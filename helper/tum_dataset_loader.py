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
import dt_utils

def load_batch_imagesV2(size,nc,dir, x,im_type):
    #This code must be modified for theano version of model
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

def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list):
    offset=0
    max_difference=0.2
    first_keys = first_list
    second_keys = second_list
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    matches=numpy.array(matches)
    return matches

def load_data(dataset,im_type):
    dir_f=dataset+im_type+'/'
    full_path=dataset+im_type+'/*.png'
    lst=glob.glob(full_path)
    n_lst = [l.replace(dir_f, '') for l in lst]
    lst = [l.replace('.png', '') for l in n_lst]
    first_list=[float(i) for i in lst]
    filename=dataset+'groundtruth.txt';
    second_list=read_file_list(filename)

    #Find closes trajectry for depth image
    matches=associate(first_list, second_list.keys())
    data_y=numpy.matrix([[float(value) for value in second_list[b][0:3]] for a,b in matches])
    data_x=[["%s%f%s" %(dir_f,a,".png")] for a,b in matches]
    rval=[(data_x),(data_y)]
    return rval

def prepare_data(step_size,data_x,data_y):
    _data_x=[]
    _data_y=[]
    overlaps = [[] for i in range(len(data_x)-1)]
    for i in range(len(data_x)):
        new_i=(i+step_size)
        if(new_i<len(data_x)):
            fImage1=data_x[i][0]
            fImage2=data_x[new_i][0]
            _data_x.append([fImage2,fImage1])
            _data_y.append(data_y[new_i,:]-data_y[i,:])
            for k in range(i,new_i):
                overlaps[k].append(i)
    rval = [(_data_x), (_data_y), (overlaps)]
    return rval

def split_test_data(dir_list,id, data_y,test_size,val_size):
    tmp_data_x = dir_list
    tmp_delta_y = data_y
    t_ind = int(round(len(tmp_delta_y) * test_size))
    v_ind = int(round(len(tmp_delta_y) * val_size))

    if(id%3==0):#mid is train
        train_inds = range(v_ind, len(tmp_delta_y) - t_ind)
        test_inds = range(len(tmp_delta_y) - t_ind, len(tmp_delta_y))
        val_inds = range(0,v_ind)

    if(id%3==1):#start is train
        train_inds = range(0, len(tmp_delta_y) - (t_ind+v_ind))
        test_inds = range(len(tmp_delta_y) - t_ind, len(tmp_delta_y))
        val_inds = range(len(tmp_delta_y) - (t_ind+v_ind),len(tmp_delta_y)- t_ind)

    if(id%3==2):#end is train
        train_inds = range(v_ind+t_ind, len(tmp_delta_y))
        test_inds = range(0 , t_ind)
        val_inds = range(t_ind , v_ind+t_ind)

    tmp_y_train = numpy.array(tmp_delta_y)[train_inds, :]
    tmp_y_test = numpy.array(tmp_delta_y)[test_inds, :]
    tmp_y_val = numpy.array(tmp_delta_y)[val_inds, :]

    tmp_y_train = tmp_y_train.reshape(len(tmp_y_train), 3)
    tmp_y_val = tmp_y_val.reshape(len(tmp_y_val), 3)

    tmp_X_train = numpy.array(tmp_data_x)[train_inds, :]
    tmp_X_test = numpy.array(tmp_data_x)[test_inds, :]
    tmp_X_val = numpy.array(tmp_data_x)[val_inds, :]


    rVal=[(tmp_X_train,tmp_y_train),(tmp_X_test,tmp_y_test),(tmp_X_val,tmp_y_val)]
    return rVal

def split_data(test_size,val_size,data_x,data_y):
        e_ind=int(round(len(data_x)*test_size))
        test_inds=range(len(data_x)-e_ind,len(data_x))
        train_inds=range(0,len(data_x)-e_ind)

        y_test=numpy.array(data_y)[test_inds,:]
        y_data=numpy.array(data_y)[train_inds,:]

        X_test=numpy.array(data_x)[test_inds,:]
        X_data=numpy.array(data_x)[train_inds,:]
        X_train, X_val, y_train, y_val= train_test_split(X_data, y_data, test_size=val_size, random_state=42)
        rval = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
        return rval

def train_test_split(X, y, test_size, random_state):
        indices=numpy.arange(len(X))
        prng = RandomState(random_state)
        prng.shuffle(indices)
        e_ind=int(round(len(X)*test_size))
        training_idx = indices[e_ind:len(X)]
        test_idx = indices[0:e_ind]
        X_training, X_test = X[training_idx,:], X[test_idx,:]
        y_training, y_test = y[training_idx,:], y[test_idx,:]
        rVal=[X_training,X_test,y_training,y_test]
        return rVal

def load_tum_data(params,id):
    dsRawData=load_data(params["dataset"][id],params["im_type"])
    dir_list=dsRawData[0]
    data_y=dsRawData[1]

    dsSplits=split_test_data(dir_list,id, data_y,params["test_size"],params["val_size"])
    raw_X_train,raw_y_train=dsSplits[0]
    raw_X_test,raw_y_test=dsSplits[1]
    raw_X_val,raw_y_val=dsSplits[2]

    X_val=[]
    X_train=[]
    X_test=[]

    y_delta_val=[]
    y_delta_train=[]
    y_delta_test=[]

    overlaps_test=[]
    overlaps_train=[]
    overlaps_val=[]

    for s in params["step_size"]:
        tmp_X_train,tmp_y_train,tmp_overlaps_train=prepare_data(s,raw_X_train,raw_y_train)
        tmp_X_test,tmp_y_test,tmp_overlaps_test=prepare_data(s,raw_X_test,raw_y_test)
        tmp_X_val,tmp_y_val,tmp_overlaps_val=prepare_data(s,raw_X_val,raw_y_val)

        if(len(X_train)==0):
            X_train=tmp_X_train
            X_test=tmp_X_test
            X_val=tmp_X_val

            y_delta_train=tmp_y_train
            y_delta_test=tmp_y_test
            y_delta_val=tmp_y_val

            if len(numpy.shape(tmp_overlaps_train))>1:
                overlaps_train=numpy.asarray(tmp_overlaps_train).reshape(len(tmp_overlaps_train))
            else:
                overlaps_train=tmp_overlaps_train

            if len(numpy.shape(tmp_overlaps_test))>1:
                overlaps_test=numpy.asarray(tmp_overlaps_test).reshape(len(tmp_overlaps_test))
            else:
                overlaps_test=tmp_overlaps_test

            if len(numpy.shape(tmp_overlaps_val))>1:
                overlaps_val=numpy.asarray(tmp_overlaps_val).reshape(len(tmp_overlaps_val))
            else:
                overlaps_val=tmp_overlaps_val

        else:
            X_train=numpy.concatenate((X_train,tmp_X_train))
            X_test=numpy.concatenate((X_test,tmp_X_test))
            X_val=numpy.concatenate((X_val,tmp_X_val))

            y_delta_train=numpy.concatenate((y_delta_train,tmp_y_train))
            y_delta_test=numpy.concatenate((y_delta_test,tmp_y_test))
            y_delta_val=numpy.concatenate((y_delta_val,tmp_y_val))

            overlaps_train=numpy.concatenate((overlaps_train,tmp_overlaps_train))
            overlaps_test=numpy.concatenate((overlaps_test,tmp_overlaps_test))
            overlaps_val=numpy.concatenate((overlaps_val,tmp_overlaps_val))

    y_delta_train=numpy.asarray(y_delta_train)
    y_delta_test=numpy.asarray(y_delta_test)
    y_delta_val=numpy.asarray(y_delta_val)

    X_train=numpy.asarray(X_train)
    X_test=numpy.asarray(X_test)
    X_val=numpy.asarray(X_val)

    y_delta_train=y_delta_train*params["multi"]
    y_delta_train=y_delta_train.reshape(len(y_delta_train),3)

    y_delta_test=y_delta_test*params["multi"]
    y_delta_test=y_delta_test.reshape(len(y_delta_test),3)


    y_delta_val=y_delta_val*params["multi"]
    y_delta_val=y_delta_val.reshape(len(y_delta_val),3)


    overlaps_train=[]
    overlaps_val=[]
    (X_train,y_train)= dt_utils.shuffle_in_unison_inplace(X_train,y_delta_train)
    (X_val,y_val)= dt_utils.shuffle_in_unison_inplace(X_val,y_delta_val)
    rval = [(X_train, y_train,overlaps_train), (X_val, y_delta_val,overlaps_val),
            (X_test, y_delta_test,overlaps_test)]
    #    model_saver.save_partitions(params["rn_id"],rval)
    return rval

def load_pairs(dataset,im_type,step_size=[]):
    offset=0
    max_difference=0.2
    random_state=42
    dsRawData=load_data(dataset,offset,max_difference)
    dir_list=dsRawData[0]
    data_y=dsRawData[1]

    X_Pairs=[]
    for s in step_size:
        tmp_X_train,tmp_y_train,tmp_overlaps_train=prepare_data(s,dir_list,data_y)

        if(len(X_Pairs)==0):
            X_Pairs=tmp_X_train
        else:
            X_Pairs=numpy.concatenate((X_Pairs,tmp_X_train))

    X_Pairs=numpy.asarray(X_Pairs)
    prng = RandomState(random_state)
    prng.shuffle(X_Pairs)

    return X_Pairs

def load_tum_data_valid(dataset,step_size,multi):
    dir_f=dataset+'/depth/'
    full_path=dataset+'/depth/*.png'
    lst=glob.glob(full_path)
    n_lst = [l.replace(dir_f, '') for l in lst]
    lst = [l.replace('.png', '') for l in n_lst]
    first_list=[float(i) for i in lst]
    filename=dataset+'groundtruth.txt'
    second_list=read_file_list(filename)

    #Find closes trajectry for depth image
    matches=associate(first_list, second_list.keys())
    data_y=numpy.matrix([[float(value) for value in second_list[b][0:3]] for a,b in matches])
    dir_list=[["%s%f%s" %(dir_f,a,".png")] for a,b in matches]
    data_x=[]
    delta_y=[]

    overlaps = [[] for i in range(len(dir_list)-1)]
    for i in range(len(dir_list)):
        fImage1=dir_list[i][0]
        new_i=(i+step_size)
        if(new_i<len(dir_list)):
            fImage2=dir_list[new_i][0]
            data_x.append([fImage1,fImage2])
            delta_y.append(data_y[i,:]-data_y[new_i,:])
            for k in range(i,new_i):
                overlaps[k].append(i)


    delta_y=numpy.asarray(delta_y)
    data_x=numpy.asarray(data_x)
    delta_y=delta_y*multi
    delta_y=delta_y.reshape(len(delta_y),3)

    rval=[data_x,delta_y,data_y,overlaps]
    return rval

def load_splits(params):
    dsRawData=load_data(params["dataset"],params["im_type"])
    data_x=dsRawData[0]
    data_y=dsRawData[1]
    data_y=data_y.reshape(len(data_y),3)
    dsSplit=split_data(params["test_size"],params["val_size"],data_x,data_y)
    X_train, y_train = dsSplit[0]
    X_val, y_val = dsSplit[1]
    X_test, y_test = dsSplit[2]
    rVal=[(X_train, y_train),(X_val, y_val),(X_test, y_test),(data_x, data_y)]
    return rVal
