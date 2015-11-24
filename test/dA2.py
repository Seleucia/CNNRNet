import os
import sys
import timeit

import dataset_loader
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.exception_verbosity = 'high'
from helper.utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        Fx=None,
        Sx=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if Fx is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.Fx = T.dmatrix(name='Fx')
            self.Sx = T.dmatrix(name='Sx')
        else:
            self.Fx = Fx
            self.Sx = Sx

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x =self.Sx
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.Fx * T.log(z) + (1 - self.Fx) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,output_folder='dA_plots'):

    size = [160, 120] #[width,height]
    batch_size=20
    dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
    X_Pairs=dataset_loader.load_pairs(dataset,step_size=[])

    n_train_batches = len(X_Pairs)
    n_train_batches /= batch_size


    Fx = T.matrix(name='Fx_input')  # the data is presented as rasterized images
    Sx = T.matrix(name='Sx_input')  # the data is presented as rasterized images

    Fx_inp = T.matrix(name='Fx_inp')  # the data is presented as rasterized images
    Sx_inp = T.matrix(name='Sx_inp')  # the data is presented as rasterized images

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        Fx=Fx,
        Sx=Sx,
        n_visible=size[0] * size[1],
        n_hidden=2000
    )

    cost, updates = da.get_cost_updates(
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [Fx_inp, Sx_inp],
        cost,
        updates=updates,
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp
        }
        ,mode="DebugMode"
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

    print "Training Started"
    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for i in xrange(n_train_batches):
            Fx = X_Pairs[i * batch_size: (i + 1) * batch_size]
            data_Fx = dataset_loader.load_batch_images(size,1, "F", Fx)
            data_Sx = dataset_loader.load_batch_images(size,1, "S", Fx)
            print("Trainin on images")
            cst=train_da(data_Fx,data_Sx)
            print("Cost:")
            print(str(cst))
            c.append(cst)

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(size[0], size[1]), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')
    os.chdir('../')


if __name__ == '__main__':
    test_dA()