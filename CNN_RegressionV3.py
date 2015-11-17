import os
import sys
import timeit
import numpy
import utils
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import dataset_loader
import model_saver
from collections import OrderedDict

# start-snippet-1

theano.config.exception_verbosity = 'high'


class OutputLayer(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True

        )

        self.p_y_given_x = T.dot(input, self.W) + self.b
        self.y_pred = self.p_y_given_x

        self.params = [self.W, self.b]
        self.input = input

    def mse(self, y):
        return T.mean((self.y_pred - y) ** 2)

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        return T.mean(T.abs_(self.y_pred - y))


class DropoutHiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, p=0.5, is_train=0):
        self.input = input
        W, b = utils.init_W_b(W, b, rng, n_in, n_out)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        output = T.maximum(lin_output, 0)

        # multiply output and drop -> in an approximation the scaling effects cancel out
        train_output = utils.drop(numpy.cast[theano.config.floatX](1. / p) * output)

        # is_train is a pseudo boolean theano variable for switching between training and prediction
        self.output = T.switch(T.neq(is_train, 0), train_output, output)

        # parameters of the model
        self.params = [self.W, self.b]


class FHiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        self.input = input

        W, b = utils.init_W_b(W, b, rng, n_in, n_out)
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = T.maximum(lin_output, 0)
        # parameters of the model
        self.params = [self.W, self.b]


class FConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        n_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        n_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                 numpy.prod(poolsize))

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
        b_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX) * numpy.cast[theano.config.floatX](0.01)

        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = theano.tensor.maximum(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'), 0)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class CNNRNet(object):
    def __init__(self,
                 rng,
                 input,
                 batch_size,
                 nc,
                 size,
                 nkerns,
                 nkern1_size, nkern2_size,
                 npool1_size, npool2_size,
                 Fx,
                 Sx):
        self.layers = []
        self.params = []
        self.dropout_layers = []

        # number of channels
        Flayer0_input = Fx.reshape((batch_size, nc, size[0], size[1]))
        Slayer0_input = Sx.reshape((batch_size, nc, size[0], size[1]))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (640-5+1 , 480-5+1) = (636, 476)
        # maxpooling reduces this further to (636/2, 476/2) = (318, 238)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], size[0], size[1])
        Flayer0 = FConvPoolLayer(
            rng,
            input=Flayer0_input,
            image_shape=(batch_size, nc, size[0], size[1]),
            filter_shape=(nkerns[0], nc, nkern1_size[0], nkern1_size[1]),
            poolsize=(npool1_size[0], npool1_size[1])
        )
        Fl0out = ((size[0] - nkern1_size[0] + 1) / npool1_size[0], (size[1] - nkern1_size[0] + 1) / npool1_size[1])

        Slayer0 = FConvPoolLayer(
            rng,
            input=Slayer0_input,
            image_shape=(batch_size, nc, size[0], size[1]),
            filter_shape=(nkerns[0], nc, nkern1_size[0], nkern1_size[1]),
            poolsize=(npool1_size[0], npool1_size[1])
        )
        Sl0out = ((size[0] - nkern1_size[0] + 1) / npool1_size[0], (size[1] - nkern1_size[0] + 1) / npool1_size[1])
        self.layers.append(Flayer0)
        self.layers.append(Slayer0)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (318-5+1, 238-5+1) = (314, 234)
        # maxpooling reduces this further to (314/2, 234/2) = (157, 117)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 157, 117)

        Flayer1 = FConvPoolLayer(
            rng,
            input=Flayer0.output,
            image_shape=(batch_size, nkerns[0]) + Fl0out,
            filter_shape=(nkerns[1], nkerns[0], nkern2_size[0], nkern2_size[1]),
            poolsize=(npool2_size[0], npool2_size[1])
        )
        Fl1out = ((Fl0out[0] - nkern2_size[0] + 1) / npool2_size[0], (Fl0out[1] - nkern2_size[0] + 1) / npool2_size[0])

        Slayer1 = FConvPoolLayer(
            rng,
            input=Slayer0.output,
            image_shape=(batch_size, nkerns[0]) + Sl0out,
            filter_shape=(nkerns[1], nkerns[0], nkern2_size[0], nkern2_size[1]),
            poolsize=(npool2_size[0], npool2_size[1])
        )
        Sl1out = ((Sl0out[0] - nkern2_size[0] + 1) / npool2_size[0], (Sl0out[1] - nkern2_size[0] + 1) / npool2_size[0])

        self.layers.append(Flayer1)
        self.layers.append(Slayer1)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 5 * 3),
        # or (500, 50 * 5 * 3) = (500, 800) with the default values.
        Flayer2_input = Flayer1.output.flatten(2)
        Slayer2_input = Slayer1.output.flatten(2)

        # construct a fully-connected relu layer
        Flayer2 = FHiddenLayer(
            rng,
            input=Flayer2_input,
            n_in=nkerns[1] * Fl1out[0] * Fl1out[1],
            n_out=250
        )

        # construct a fully-connected relu layer
        Slayer2 = FHiddenLayer(
            rng,
            input=Slayer2_input,
            n_in=nkerns[1] * Sl1out[0] * Sl1out[1],
            n_out=250
        )

        self.layers.append(Flayer2)
        self.layers.append(Slayer2)

        # construct a fully-connected sigmoidal layer
        layer3 = FHiddenLayer(
            rng,
            input=T.concatenate([Flayer2.output, Slayer2.output], axis=1),
            n_in=500,
            n_out=500
        )
        self.layers.append(layer3)

        # construct a fully-connected sigmoidal layer
        layer4 = FHiddenLayer(
            rng,
            input=layer3.output,
            n_in=500,
            n_out=500
        )
        self.layers.append(layer4)

        layerOutput = OutputLayer(input=layer4.output, n_in=500, n_out=3)
        self.layers.append(layerOutput)

        for layer in zip(reversed(self.layers)):
            self.params.append(layer[0].W)
            self.params.append(layer[0].b)
        # create a list of all model parameters to be fit by gradient descent
        #self.params = layerOutput.params + layer4.params + layer3.params + Flayer2.params + Slayer2.params + Flayer1.params + Slayer1.params + Flayer0.params + Slayer0.params
        # Use the negative log likelihood of the logistic regression layer as
        # the objective.

        L1 = shared(0.)
        for param in self.params:
            L1 += (T.sum(abs(param[0]))+T.sum(abs(param[1])))
        self.L1=L1

        L2_sqr = shared(0.)
        for param in self.params:
            L2_sqr += (T.sum(param[0] ** 2)+T.sum(param[1] ** 2))
        self.L2_sqr=L2_sqr



        self.y_pred = self.layers[-1].y_pred
        self.mse = self.layers[-1].mse
        self.errors = self.layers[-1].errors
    def set_params(self,params):
        dir =0
        for param in reversed(numpy.reshape(params,(9,2))):
            self.layers[dir].params[0]=param[0]
            self.layers[dir].params[1]=param[1]
            self.layers[dir].W=param[0]
            self.layers[dir].b=param[1]

            dir=dir+1

        print "okkk"

def train_model():
    dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
    rng = numpy.random.RandomState(23455)
    # size = [480,640] orijinal size,[height,width]
    rn_id=1
    size = [160, 120] #[width,height]
    nc = 1  # number of channcels
    nkerns = [20, 50]
    nkern1_size = [5, 5]
    nkern2_size = [5, 5]

    npool1_size = [2, 2]
    npool2_size = [2, 2]

    batch_size = 30

    multi = 10

    learning_rate = 0.0005
    n_epochs = 400

    initial_learning_rate = 0.0005
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    n_epochs = 3000
    learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))

    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 500
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}

    lambda_1 = 0.01  # regulizer param
    lambda_2 = 0.01

    datasets = dataset_loader.load_tum_dataV2(dataset,rn_id,multi)

    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(X_train)
    n_valid_batches = len(X_val)
    n_test_batches = len(X_test)

    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    epoch = T.scalar()
    Fx = T.matrix(name='Fx_input')  # the data is presented as rasterized images
    Sx = T.matrix(name='Sx_input')  # the data is presented as rasterized images
    y = T.matrix('y')  # the output are presented as matrix 1*3.

    Fx_inp = T.matrix(name='Fx_inp')  # the data is presented as rasterized images
    Sx_inp = T.matrix(name='Sx_inp')  # the data is presented as rasterized images
    y_inp = T.matrix('y_inp')

    print '... building the model'

    cnnr = CNNRNet(rng, input, batch_size, nc, size, nkerns,
                   nkern1_size, nkern2_size,
                   npool1_size, npool2_size,
                   Fx, Sx)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [Fx_inp, Sx_inp, y_inp],
        cnnr.errors(y),
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp,
            y: y_inp,
        }

    )

    validate_model = theano.function(
        [Fx_inp, Sx_inp, y_inp],
        cnnr.errors(y),
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp,
            y: y_inp,
        }
    )

    cost = cnnr.mse(y) + lambda_1 * cnnr.L1 + lambda_2 * cnnr.L2_sqr

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in cnnr.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # ... and allocate mmeory for momentum'd versions of the gradient
    gparams_mom = []
    for param in cnnr.params:
        gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                               dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # Compute momentum for the current epoch
    mom = mom_start * (1.0 - epoch / mom_epoch_interval) + mom_end * (epoch / mom_epoch_interval) if T.lt(epoch,
                                                                                                          mom_epoch_interval) else mom_end

    # Update the step direction using momentum
    updates = OrderedDict()

    for gparam_mom, gparam in zip(gparams_mom, gparams):
        # Misha Denil's original version
        # updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam

        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(cnnr.params, gparams_mom):
        # Misha Denil's original version
        # stepped_param = param - learning_rate * updates[gparam_mom]

        # since we have included learning_rate in gparam_mom, we don't need it
        # here
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            # squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            # scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            # updates[param] = stepped_param * scale

            # constrain the norms of the COLUMNs of the weight, according to
            # https://github.com/BVLC/caffe/issues/109
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param

    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = cost

    train_model = theano.function(
        [Fx_inp, Sx_inp, y_inp, epoch],
        outputs=output,
        updates=updates,
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp,
            y: y_inp,
        }

    )
    # create a function to compute the mistakes that are made by the model
    predict_model = theano.function(
        [Fx_inp, Sx_inp],
        cnnr.y_pred,
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp,
        }
    )
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                          updates={learning_rate: learning_rate * learning_rate_decay})
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    epoch_counter = 0

    while (epoch_counter < n_epochs) and (not done_looping):
        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch_counter - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print 'training @ iter = ', iter

            Fx = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx)
            data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx)
            data_y = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            cost_ij = train_model(data_Fx, data_Sx, data_y, epoch)
            # model_saver.save_model(epoch % 3, params)
            print('epoch %i, minibatch %i/%i, training cost %f ' %
                  (epoch_counter, minibatch_index + 1, n_train_batches,
                   cost_ij))

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = 0
                for i in xrange(n_valid_batches):
                    Fx = X_val[i * batch_size: (i + 1) * batch_size]
                    data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx)
                    data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx)
                    data_y = y_val[i * batch_size: (i + 1) * batch_size]
                    validation_losses = validation_losses + validate_model(data_Fx, data_Sx, data_y)

                this_validation_loss = validation_losses / n_valid_batches
                new_learning_rate = decay_learning_rate()

                print('epoch %i, minibatch %i/%i, learning_rate %f validation error %f %%' %
                      (epoch_counter, minibatch_index + 1, n_train_batches,
                       learning_rate.get_value(borrow=True),
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = 0
                    for i in xrange(n_test_batches):
                        Fx = X_test[i * batch_size: (i + 1) * batch_size]
                        data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx)
                        data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx)
                        data_y = y_test[i * batch_size: (i + 1) * batch_size]
                        err= test_model(data_Fx, data_Sx, data_y)
                        test_losses = test_losses + err
                        if(i%10==0):
                            store=[]
                            ypred= predict_model(data_Fx, data_Sx)
                            store.append(Fx)
                            store.append(ypred)
                            store.append(data_y)
                            model_saver.save_garb(store)
                            print("Iteration saved %i, err %f"%(i,err))

                    test_score = test_losses / n_test_batches
                    ext=str(rn_id)+"_"+str(epoch_counter % 3)
                    model_saver.save_model(ext, cnnr.params)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch_counter, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


train_model()
