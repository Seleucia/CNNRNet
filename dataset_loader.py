import cPickle
import gzip
import os
import numpy
import theano
import theano.tensor as T
import glob
from theano import config
from sklearn.cross_validation import train_test_split
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

def load_batch_images(size,nc,dir, x):
    #We should modify this function to load images with different number of channels
    fl_size=size[0]*size[1]
    m_size = (len(x), fl_size)
    data_x = numpy.empty(m_size, float)
    i = 0
    for (dImg1, dImg2) in x:
        dImg=""
        if dir=="F":
            dImg=dImg1
        else:
            dImg=dImg2
        img = Image.open(dImg)
        img=img.resize(size)
        arr1= numpy.array(img,float)/5000
        v_1 = numpy.transpose(numpy.reshape(arr1, (fl_size, 1)))
        data_x[i, :] = v_1
        i = i + 1;
    return data_x

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
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

def load_tum_dataV2(dataset,rn_id,multi):
    offset=0
    max_difference=0.2
    test_size=0.20
    val_size=0.20

    dir_f=dataset+'/depth/'
    full_path=dataset+'/depth/*.png'
    lst=glob.glob(full_path)
    n_lst = [l.replace(dir_f, '') for l in lst]
    lst = [l.replace('.png', '') for l in n_lst]
    first_list=[float(i) for i in lst]
    filename=dataset+'/groundtruth.txt';
    second_list=read_file_list(filename)

    #Find closes trajectry for depth image
    matches=associate(first_list, second_list.keys(),offset,max_difference)
    data_y=numpy.matrix([[float(value) for value in second_list[b][0:3]] for a,b in matches])
    dir_list=[["%s%f%s" %(dir_f,a,".png")] for a,b in matches]
    data_x=[]
    delta_y=[]
    X_test=[]
    y_test=[]
    step_size=[2,5,10,13,15,18,20,23,25,27,30,32,38,35,40,45,50]
    for s in step_size:
        i=0
        tmp_data_x=[]
        tmp_delta_y=[]
        for i in range(len(dir_list)):
            fImage1=dir_list[i][0]
            new_i=(i+s)
            if(new_i<len(dir_list)):
                fImage2=dir_list[new_i][0]
                tmp_data_x.append([fImage1,fImage2])
                tmp_delta_y.append(data_y[i,:]-data_y[new_i,:])

        e_ind=int(round(len(tmp_delta_y)*test_size))
        test_inds=range(len(tmp_delta_y)-e_ind,len(tmp_delta_y))
        train_inds=range(0,len(tmp_delta_y)-e_ind)
        tmp_y_test=numpy.array(tmp_delta_y)[test_inds,:]
        tmp_X_test=numpy.array(tmp_data_x)[test_inds,:]
        tmp_train_y=numpy.array(tmp_delta_y)[train_inds,:]
        tmp_train_x=numpy.array(tmp_data_x)[train_inds,:]

        tmp_train_y=tmp_train_y.reshape(len(tmp_train_y),3)
        tmp_train_x=tmp_train_x.reshape(len(tmp_train_x),2)
        if(len(data_x)==0):
            data_x=tmp_train_x
            delta_y=tmp_train_y
            X_test=tmp_X_test
            y_test=tmp_y_test
        else:
            data_x=numpy.concatenate((data_x,tmp_train_x))
            delta_y=numpy.concatenate((delta_y,tmp_train_y))
            X_test=numpy.concatenate((X_test,tmp_X_test))
            y_test=numpy.concatenate((y_test,tmp_y_test))

    y_test=numpy.asarray(y_test)
    X_test=numpy.asarray(X_test)
    y_test=y_test*multi
    y_test=y_test.reshape(len(y_test),3)

    delta_y=numpy.asarray(delta_y)
    data_x=numpy.asarray(data_x)
    delta_y=delta_y*multi
    delta_y=delta_y.reshape(len(delta_y),3)

    print("Data loaded split started")
    X_train, X_val, y_train, y_val= train_test_split(data_x, delta_y, test_size=val_size, random_state=42)
    del data_x
    del delta_y

    (X_train,y_train)= shuffle_in_unison_inplace(numpy.asarray(X_train),numpy.asarray(y_train))
    rval = [(X_train, y_train), (X_val, y_val),
            (X_test, y_test)]
    model_saver.save_partitions(rn_id,rval)
    return rval

def load_tum_data_valid(dataset,step_size,multi):

    offset=0
    max_difference=0.2


    dir_f=dataset+'/depth/'
    full_path=dataset+'/depth/*.png'
    lst=glob.glob(full_path)
    n_lst = [l.replace(dir_f, '') for l in lst]
    lst = [l.replace('.png', '') for l in n_lst]
    first_list=[float(i) for i in lst]
    filename=dataset+'/groundtruth.txt'
    second_list=read_file_list(filename)

    #Find closes trajectry for depth image
    matches=associate(first_list, second_list.keys(),offset,max_difference)
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

def save_batches(datasets,batch_size,fl_size):
        print("Batch saving started")
        X_train, y_train = datasets[0]
        X_val, y_val = datasets[1]
        X_test, y_test = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = len(X_train)
        n_valid_batches = len(X_val)
        n_test_batches = len(X_test)
        n_total=n_train_batches+n_valid_batches+n_test_batches
        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size
        for index in xrange(n_train_batches):
            x=X_train[index * batch_size: (index + 1) * batch_size]
            data_x=load_batch_images(batch_size, fl_size, x)
            data_y=y_train[index * batch_size: (index + 1) * batch_size]
            (x,y)= shared_dataset(data_x,data_y)
            pickle.dump( x+y, open("batches/tr_"+str(index)+".p", "wb" ) )

        for index in xrange(n_valid_batches):
            x=X_val[index * batch_size: (index + 1) * batch_size]
            data_x=load_batch_images(batch_size, fl_size, x)
            data_y=y_val[index * batch_size: (index + 1) * batch_size]
            (x,y)= shared_dataset(data_x,data_y)
            pickle.dump( x+y, open("batches/val_"+str(index)+".p", "wb" ) )

        for index in xrange(n_test_batches):
            x=X_test[index * batch_size: (index + 1) * batch_size]
            data_x=load_batch_images(batch_size, fl_size, x)
            data_y=y_test[index * batch_size: (index + 1) * batch_size]
            (x,y)= shared_dataset(data_x,data_y)
            pickle.dump( x+y, open("batches/te_"+str(index)+".p", "wb" ) )

        print("Batch saving ended")