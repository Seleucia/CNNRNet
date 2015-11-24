import numpy
import theano
import theano.tensor as T


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

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type,y.ndim, 'y_pred', self.y_pred.type,self.y_pred.ndim)
            )
        return T.mean(T.abs_(self.y_pred - y))

    def mse(self, y):
        return T.mean((self.y_pred - y) ** 2)
