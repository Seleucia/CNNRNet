import numpy
import utils
import theano
import theano.tensor as T

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
