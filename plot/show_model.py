from __future__ import absolute_import
from __future__ import print_function
import pylab as pl
import matplotlib.cm as cm
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import helper.config as config
import plot_data
import models.model_provider as mp
import theano
import numpy.ma as ma

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

#pl.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1))

params=config.get_params()
params['model_name']="more_noreg_data2_46_gray.hdf5"
params['model']="kcnnr"

model=mp.get_model_pretrained(params)


for i in range(2):
   print ("Lef or right: %s th"%(i))
   for k in range(3):
      print ("Layer param:of %s th"%(i))
      # Visualize weights
      W=np.squeeze(model.layers[0].layers[i].layers[k*3].W.get_value(borrow=True))
      print("W shape : ", W.shape)
      pl.figure(figsize=(15, 15))
      pl.title('conv1 weights (i,k)(%s,%s)'%(i,k))
      if(len(W.shape)>3):
         X=W.reshape(W.sahpe[0],20,30)
         mosaic=make_mosaic(X, 6, 6)
         nice_imshow(pl.gca(), mosaic, cmap=cm.binary)
         pl.show()
         # for z in range(W.shape[0]):
         #    mosaic=make_mosaic(W[i], 7, 7)
         #    nice_imshow(pl.gca(), mosaic, cmap=cm.binary)
         #    pl.show()
         #    if z>5:
         #       break
      else:
            mosaic=make_mosaic(W, 7,7)
            nice_imshow(pl.gca(), mosaic, cmap=cm.binary)
            pl.show()

