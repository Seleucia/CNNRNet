from theano import function
import theano.tensor as T
from theano import shared
import  numpy
import  theano

n_out=10
state = shared(0)
b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
b = theano.shared(b_values,name='b', borrow=True)

inc = T.iscalar('inc')
fn_of_state = state * 2 + inc

foo = T.scalar(dtype=state.dtype)

skip_shared = function([inc], fn_of_state)
state.set_value(10)
b_values = numpy.ones((n_out,), dtype=theano.config.floatX)
b.set_value(b_values)
print( skip_shared(1) )
