import os
import sys
import timeit

import dataset_loader
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from helper import model_saver

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


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, ):
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = T.maximum(lin_output, 0)
        # parameters of the model
        self.params = [self.W, self.b]


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

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


def train_model():
    rng = numpy.random.RandomState(23455)
    #size = [480,640] orijinal size
    size = [120,160]

    nkerns = [20, 50]
    nkern1_size = [5, 5]
    nkern2_size = [5, 5]

    npool1_size = [2, 2]
    npool2_size = [2, 2]

    batch_size = 30
    fl_size = size[0] * size[1]
    multi = 100
    learning_rate = 0.001
    n_epochs = 400

    datasets = dataset_loader.load_tum_dataV2(size, multi)

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

    x = T.tensor3(name='input')  # the data is presented as rasterized images
    y = T.matrix('y')  # the output are presented as matrix 1*3.

    x_inp = T.tensor3(name='x_inp')  # the data is presented as rasterized images
    y_inp = T.matrix('y_inp')

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # number of channels
    layer0_input = x.reshape((batch_size, 2, size[0], size[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (640-5+1 , 480-5+1) = (636, 476)
    # maxpooling reduces this further to (636/2, 476/2) = (318, 238)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], size[0], size[1])
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 2, size[0], size[1]),
        filter_shape=(nkerns[0], 2, nkern1_size[0], nkern1_size[1]),
        poolsize=(npool1_size[0], npool1_size[1])
    )
    l0out = ((size[0] - nkern1_size[0] + 1) / npool1_size[0], (size[1] - nkern1_size[0] + 1) / npool1_size[1])

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (318-5+1, 238-5+1) = (314, 234)
    # maxpooling reduces this further to (314/2, 234/2) = (157, 117)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 157, 117)

    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0]) + l0out,
        filter_shape=(nkerns[1], nkerns[0], nkern2_size[0], nkern2_size[1]),
        poolsize=(npool2_size[0], npool2_size[1])
    )
    l2out = ((l0out[0] - nkern2_size[0] + 1) / npool2_size[0], (l0out[1] - nkern2_size[0] + 1) / npool2_size[0])

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 5 * 3),
    # or (500, 50 * 5 * 3) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * l2out[0] * l2out[1],
        n_out=500
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = OutputLayer(input=layer2.output, n_in=500, n_out=3)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x_inp, y_inp],
        layer3.errors(y),
        givens={
            x: x_inp,
            y: y_inp,
        }

    )

    validate_model = theano.function(
        [x_inp, y_inp],
        layer3.errors(y),
        givens={
            x: x_inp,
            y: y_inp,
        }
    )


    #Creat cost
    L1 = (abs(layer3.W).sum()) + (abs(layer2.W).sum()) + (abs(layer1.W).sum()) + (abs(layer0.W).sum())
    L2_sqr = ((layer3.W ** 2).sum()) + ((layer1.W ** 2).sum()) + ((layer1.W ** 2).sum()) + ((layer0.W ** 2).sum())
    lambda_1 = 0.1
    lambda_2 = 0.1
    cost = layer3.mse(y) + lambda_1 * L1 + lambda_2 * L2_sqr

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
        ]

    train_model = theano.function(
        [x_inp, y_inp],
        cost,
        updates=updates,
        givens={
            x: x_inp,
            y: y_inp,
        }
    )
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

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter

            x = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            data_x = dataset_loader.load_batch_imagesV2(size, x)
            data_y = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            cost_ij = train_model(data_x, data_y)
            #model_saver.save_model(epoch % 3, params)
            print('epoch %i, minibatch %i/%i, training cost %f ' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   cost_ij))

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = 0
                for i in xrange(n_valid_batches):
                    x = X_val[i * batch_size: (i + 1) * batch_size]
                    data_x = dataset_loader.load_batch_imagesV2(size, x)
                    data_y = y_val[i * batch_size: (i + 1) * batch_size]
                    validation_losses = validation_losses + validate_model(data_x, data_y)

                this_validation_loss = validation_losses / n_valid_batches

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
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
                        x = X_test[i * batch_size: (i + 1) * batch_size]
                        data_x = dataset_loader.load_batch_imagesV2(size,  x)
                        data_y = y_test[i * batch_size: (i + 1) * batch_size]
                        test_losses = test_losses + validate_model(data_x, data_y)

                    test_score = test_losses / n_valid_batches
                    model_saver.save_model(epoch % 3, params)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
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
