import tables
import os
import numpy
import theano
import plot.plot_data as pd
import glob
from theano import config
import theano.tensor as T
from PIL import Image
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from theano.tensor.extra_ops import to_one_hot

def shared_dataset(data_xx, data_yy, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_xx,
                                               dtype=theano.config.floatX),config.floatX,
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_yy,
                                               dtype=theano.config.floatX),config.floatX,
                                 borrow=borrow)
        return shared_x, shared_y

def load_batch_images(params, direction, x, patch_loc,map_loc):
    #We should modify this function to load images with different number of channels
    size=params["size"]
    im_type=params["im_type"]
    patch_use=params['patch_use']

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

    if(im_type=="hha_depth_fc6"):
        normalizer=47.2940864563
        sbt=params["hha_depth_fc6_mean"]

    if(im_type=="rgb_conv4_2"):
        normalizer=8010.15771484
        sbt=params["rgb_conv4_2_mean"]

    if(im_type=="rgb_conv5_2"):
        normalizer=1042.07519531
        sbt=params["rgb_conv5_2_mean"]

    im_order=0
    if(direction=="S"):
        im_order=1
    map_arg=[(direction, im_type, normalizer, patch_loc,map_loc, patch_use, sbt, size, im[im_order]) for im in x]
    pool_img = ThreadPool(params["n_procc"])
    pool_img.daemon=True
    results = pool_img.map(load_image_wrapper,map_arg)
    pool_img.close()
    pool_img.join()
    batch_l=convert_set(results,im_type)
    return numpy.array(batch_l)

def convert_to_class(data_y):
    tmp=numpy.zeros_like(data_y)
    tmp[data_y > 0] = 2
    tmp[data_y == 0] = 1
    tmp[data_y < 0] = 0
    numpy_1_of_k_1(tmp, 3)
    res=theano.tensor.extra_ops.to_one_hot(tmp,3)
    return restmp

def class_idx_seq_to_1_of_k(seq, num_classes, dtype="float32"):
  shape = [seq.shape[i] for i in range(seq.ndim)] + [num_classes]
  eye = T.eye(num_classes, dtype=dtype)
  m = eye[T.cast(seq, 'int32')].reshape(shape)
  return m

def numpy_1_of_k_1(seq, num_classes):
    num_frames = len(seq)
    m = numpy.zeros((num_frames, num_classes))
    m[numpy.arange(num_frames), seq] = 1
    return m


def convert_set(res,im_type):
    batch_l = []
    i = 0
    for arr1 in res:
        l = []
        l.append([])
        l[0] = arr1
        n_l = numpy.array(l)
        batch_l.append([])
        batch_l[i] = n_l
        i += 1

    if (im_type.find("fc") > 0) or (im_type.find("conv")>0):
        batch_l=numpy.squeeze(batch_l)
    return batch_l

def load_image(direction, im_type, normalizer, patch_loc,map_loc, patch_use, sbt, size, dImg):
    if (im_type.find("fc") > 0):
            arr1 = numpy.load(dImg)
    elif (im_type.find("conv")>0):
        arr1=read_hdf(dImg)
        arr1=arr1[map_loc]
        arr1= numpy.squeeze(arr1.reshape(arr1.shape[0]*arr1.shape[1]))
    else:
            img = Image.open(dImg)
            if (patch_use == 1):
                img = img.crop(patch_loc)
            else:
                img = img.resize(size)
            arr1 = numpy.array(img, theano.config.floatX)

    if (im_type == "depth"):
        arr2 = numpy.zeros_like(arr1)
        arr2[arr1.nonzero()] = sbt
        arr1 = (arr1 - arr2) / normalizer
    else:
        arr1 = (arr1 - sbt) / normalizer
    return arr1

def load_image_wrapper(args):
   return load_image(*args)

def asyn_load_batch_images(args):
    params=args[0][0]
    pool = Pool(params["n_pool"])
    pool.daemon=True
    results = pool.map(load_batch_wrapper,args)
    pool.close()
    pool.join()
    return results

def load_batch_wrapper(args):
   return load_batch_images(*args)

def write_features(ndarr, fl_ls,parent_dir):
    for i in range(ndarr.shape[1]):
        f=ndarr[:,i]
        f_name=os.path.basename(fl_ls[i])
        full_path=parent_dir+f_name.replace(".png","")
        numpy.save(full_path,f)

def write_mid_features(data,parent_dir, fl_ls):
    for i in range(data.shape[0]):
        f=data[i]
        f_name=os.path.basename(fl_ls[i])
        full_path=parent_dir+f_name.replace(".png","")
        save_hdf(full_path,f)

def save_hdf(f_name,all_data):
    f = tables.openFile(f_name+'.hdf', 'w')
    atom = tables.Atom.from_dtype(all_data.dtype)
    filters = tables.Filters(complib='blosc', complevel=9,least_significant_digit=2)
    ds = f.createCArray(f.root, 'data', atom, all_data.shape, filters=filters)
    # save w/o compressive filter
    #ds = f.createCArray(f.root, 'all_data', atom, all_data.shape)
    ds[:] = all_data
    f.close()

def read_hdf(f_name):
    f = tables.openFile(f_name, 'r')
    data=f.root.data[:]
    f.close()
    return data

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def check_missin_values(params):
    orijinal_img="depth"
    for dir in params["dataset"]:
        if (dir == -1):
            continue
        rgb_dir= dir[0] + orijinal_img + '/*.png'
        path_imgs =glob.glob(rgb_dir)
        path_imgs=sorted(path_imgs)
        ms=numpy.zeros((len(path_imgs),1))
        i=0
        for dImg in path_imgs:
            img = Image.open(dImg)
            arr1= numpy.array(img,theano.config.floatX)
            p=(arr1.size-numpy.count_nonzero(arr1))/float(arr1.size)
            ms[i]=p
            i+=1
        pd.plot_ms(ms,dir[0],os.path.basename(os.path.normpath(dir[0])))
