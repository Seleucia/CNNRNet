import numpy
import theano
import theano.tensor as T
import os
import glob
from PIL import Image

def init_W_b(W, b, rng, n_in, n_out):
    # for a discussion of the initialization, see
    # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
    if W is None:
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6./(n_in + n_out)),
                high=numpy.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
    if b is None:
        b_values = numpy.ones((n_out,), dtype=theano.config.floatX) * numpy.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b


def init_CNNW_b(W, b, rng, n_in, n_out,fshape):
    # for a discussion of the initialization, see
    # https://plus.google.com/+EricBattenberg/posts/f3tPKjo7LFa
    if W is None:
        W_values = numpy.asarray(
            rng.uniform(
                low=-numpy.sqrt(6./(n_in + n_out)),
                high=numpy.sqrt(6./(n_in + n_out)),
                size=(n_in, n_out)
                ),
            dtype=theano.config.floatX
        )
        W = theano.shared(value=W_values, name='W', borrow=True)

    # init biases to positive values, so we should be initially in the linear regime of the linear rectified function
    if b is None:
        b_values = numpy.ones((fshape,), dtype=theano.config.floatX) * numpy.cast[theano.config.floatX](0.01)
        b = theano.shared(value=b_values, name='b', borrow=True)
    return W, b

rng = numpy.random.RandomState(1234)
srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))

def drop(input, p=0.5, rng=rng):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.

    """
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

def rescale_weights(params, incoming_max):
    incoming_max = numpy.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * numpy.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)

def up_sample(overlaps,data_y,step_size):
    data_yy=[]
    for index in range(len(overlaps)):
        value= sum(numpy.asarray(data_y)[overlaps[index]]) / (step_size * len(overlaps[index]))
        data_yy.append(value)
    return numpy.asarray(data_yy)

def convert_to_grayscale():
    dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
    im_type='rgb'
    im_type_to='gray'
    new_dir=dataset+im_type_to+"/"
    full_path=dataset+'/'+im_type+'/*.png'
    lst=glob.glob(full_path)
    for f in lst:
        img = Image.open(f).convert('L')
        img.save(new_dir+os.path.basename(f))


