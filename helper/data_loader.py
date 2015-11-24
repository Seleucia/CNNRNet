import tum_dataset_loader as tumdata
import numpy as np
import config
import dt_utils

def load_data(params):
    X_train=[]
    X_val=[]
    X_test=[]
    Y_train=[]
    Y_test=[]
    Y_val=[]
    Overlaps_train=[]
    Overlaps_val=[]
    Overlaps_test=[]
    for dir in params["dataset"]:
        datasets = tumdata.load_tum_data(params,dir)
        x_train, y_train,overlaps_train = datasets[0]
        x_val, y_val,overlaps_val = datasets[1]
        x_test, y_test,overlaps_test = datasets[2]
        if(len(X_train)==0):
            X_train=np.array(x_train)
            X_val=np.array(x_val)
            X_test=np.array(x_test)
            Y_train=np.array(y_train)
            Y_test=np.array(y_test)
            Y_val=np.array(y_val)
            Overlaps_train=np.array(overlaps_train)
            Overlaps_val=np.array(overlaps_val)
            Overlaps_test=np.array(overlaps_test)
        else:
            X_train=np.concatenate((X_train,x_train),axis=0)
            X_val=np.concatenate((X_val,x_val),axis=0)
            X_test=np.concatenate((X_test,x_test),axis=0)
            Y_train=np.concatenate((Y_train,y_train),axis=0)
            Y_val=np.concatenate((Y_val,y_val),axis=0)
            Y_test=np.concatenate((Y_test,y_test),axis=0)
            Overlaps_train=np.concatenate((Overlaps_train,overlaps_train),axis=0)
            Overlaps_val=np.concatenate((Overlaps_val,overlaps_val),axis=0)
            Overlaps_test=np.concatenate((Overlaps_test,overlaps_test),axis=0)

    X_train,Y_train=dt_utils.shuffle_in_unison_inplace(X_train,Y_train)
    X_val,Y_val=dt_utils.shuffle_in_unison_inplace(X_val,Y_val)
    rval = [(X_train, Y_train,Overlaps_train), (X_val, Y_val,Overlaps_val),
            (X_test, Y_test,Overlaps_test)]
    return rval


params=config.get_params("home")
dataset=load_data(params)