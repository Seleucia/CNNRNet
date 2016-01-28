import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

theano.config.floatX = 'float32'
#theano.config.optimizer='fast_compile'
n=T.fscalar()

def minus(x, y):
    return x - y


def add(x, y):
    return x + y

scan_ints_expr, updates = theano.scan(add,
                                      sequences=T.arange(n),
                                      outputs_info=np.float64(0))
f_scan_integers = theano.function([n], scan_ints_expr)

print f_scan_integers(10)


another_alternating_expr, updates = theano.scan(
        minus,
        sequences=None,
        outputs_info=[dict(initial=np.int32([1, 1]), taps=[-1, -2])],
        n_steps=10)

f_alternating2 = theano.function([], another_alternating_expr,allow_input_downcast=True)
f_alternating2()