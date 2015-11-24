import cPickle
import gzip
import os
import numpy
import theano
from numpy.random import RandomState
import glob
from theano import config
from PIL import Image
import pickle
import model_saver

def shared_dataset(data_xx, data_yy, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_xx,
                                               dtype=theano.config.floatX),config.floatX,
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_yy,
                                               dtype=theano.config.floatX),config.floatX,
                                 borrow=borrow)
        return shared_x, shared_y

def load_batch_images(size,nc,dir, x,im_type):
    #We should modify this function to load images with different number of channels
    fl_size=size[0]*size[1]
    m_size = (len(x), fl_size)
    data_x = numpy.empty(m_size, theano.config.floatX)
    i = 0
    normalizer=5000
    img_arr=[]
    if(im_type=="gray"):
        normalizer=255
    batch_l=[]
    for (dImg1, dImg2) in x:
        dImg=""
        if dir=="F":
            dImg=dImg1
        else:
            dImg=dImg2
        img = Image.open(dImg)
        img=img.resize(size)
        arr1= numpy.array(img,theano.config.floatX)/normalizer
        l=[]
        l.append([])
        l[0]=arr1
        n_l=numpy.array(l)
        batch_l.append([])
        batch_l[i]=n_l
        i+=1
    return numpy.array(batch_l)


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]
