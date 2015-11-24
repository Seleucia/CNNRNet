import os

def get_params(location):
    params={}
    params["rn_id"]=1 #running id
    # early-stopping parameters
    params['patience']= 10000  # look as this many examples regardless
    params['patience_increase']=2  # wait this much longer when a new best is
    params['improvement_threshold']=0.995  # a relative improvement of this much is

    # learning parameters
    params['momentum']=0.9    # the params for momentum
    params['initial_learning_rate']=0.0001
    params['learning_rate_decay']= 0.998
    params['squared_filter_length_limit']=15.0
    params['batch_size']=10
    params['n_epochs']=3000

    # dataset parameters
    if(location=="home"):
        params['dataset']="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
    if(location=="tesla"):
        params['dataset']="/home/cmp/projects/data/rgbd_dataset_freiburg3_large_cabinet/" #test computer
    if(location=="std"):
        params['dataset']="/home/cmp/projects/data/rgbd_dataset_freiburg3_large_cabinet/" #test computer

    params['im_type']="gray"
    params['step_size']=[1,2,5,7,10,12,13,14,15,16,18,20,21,23,24,25]
    #params['step_size']=[10]
    params['size']=[160, 120] #[width,height]
    params['nc']=1 #number of dimensions
    params['multi']=10 #ground truth location differences will be multiplied with this number
    params['test_size']=0.20 #Test size
    params['val_size']=0.20 #Test size

    # c an Pooling parameters
    params['kern_mat']=[(5, 5), (5, 5)] #shape of kernel
    params['nkerns']= [40, 30,20,20] #number of kernel
    params['pool_mat']=  [(2, 2), (2, 2)] #shape of pooling

    # os
    wd=os.getcwd()
    params['wd']=wd
    params['models']=wd+"/models/"
    params['logs']=wd+"/logs/"
    params['model_name']=wd+"models/1_2.h5"
    return params

