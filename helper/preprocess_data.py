import numpy
import theano
import theano.tensor as T
import os
import glob
from PIL import Image
from shutil import rmtree
import numpy as np
import config

def depth_meansubtract(params):
    for dir in params["dataset"]:
        if dir ==-1:
            continue
        normalizer=52492
        sbt=params["depth_mean"]
        im_type='depth'
        im_type_to='mean_depth'
        new_dir=dir[0]+im_type_to+"/"
        if os.path.exists(new_dir):
           rmtree(new_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            full_path=dir[0]+'/'+im_type+'/*.png'
            lst=glob.glob(full_path)
            for f in lst:
                img = Image.open(f)
                arr1= numpy.array(img,theano.config.floatX)
                arr2=np.zeros_like(arr1)
                arr2[arr1.nonzero()]=sbt
                arr1=(arr1-arr2)/normalizer
                f=new_dir+os.path.basename(f).replace(".png","")
                np.save(f,arr1)
            print("data set converted %s"%(dir[0]))
        else:
            print("data set has already proccessed %s"%(dir[0]))
    print "Depth data proccessing completed"

params=config.get_params()
depth_meansubtract(params)