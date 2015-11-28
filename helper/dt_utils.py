import cPickle
import gzip
import os
import numpy
import theano
from numpy.random import RandomState
import glob
from theano import config
from PIL import Image

def shared_dataset(data_xx, data_yy, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_xx,
                                               dtype=theano.config.floatX),config.floatX,
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_yy,
                                               dtype=theano.config.floatX),config.floatX,
                                 borrow=borrow)
        return shared_x, shared_y

def load_batch_images(params,dir, x):
    #We should modify this function to load images with different number of channels
    size=params["size"]
    nc=params["nc"]
    im_type=params["im_type"]
    fl_size=size[0]*size[1]
    m_size = (len(x), fl_size)
    data_x = numpy.empty(m_size, theano.config.floatX)
    img_arr=[]

    sbt=1
    if(im_type=="depth"):
        normalizer=52492
        sbt=params["depth_mean"]

    if(im_type=="pre_depth"):
        normalizer=52492
        sbt=params["pre_depth_mean"]

    if(im_type=="gray"):
        normalizer=255
        sbt=params["gray_mean"]
    if(im_type=="rgb"):
        normalizer=255
        sbt=params["rgb_mean"]

    batch_l=[]
    i = 0
    for (dImg1, dImg2) in x:
        dImg=""
        if dir=="F":
            dImg=dImg1
        else:
            dImg=dImg2
        img = Image.open(dImg)
        img=img.resize(size)
        arr1= numpy.array(img,theano.config.floatX)
        arr1=(arr1-sbt)/normalizer
        l=[]
        l.append([])
        l[0]=arr1
        n_l=numpy.array(l)
        batch_l.append([])
        batch_l[i]=n_l
        i+=1
    return numpy.array(batch_l)

def write_features(ndarr, fl_ls,parent_dir):
    for i in range(ndarr.shape[1]):
        f=ndarr[:,i]
        f_name=os.path.basename(fl_ls[i])
        full_path=parent_dir+f_name.replace(".png","")
        numpy.save(full_path,f)


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]
