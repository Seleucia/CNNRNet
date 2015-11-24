import theano.tensor as T

from helper import utils


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        self.input = input

        W, b = utils.init_W_b(W, b, rng, n_in, n_out)
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = T.maximum(lin_output, 0)
        # parameters of the model
        self.params = [self.W, self.b]
