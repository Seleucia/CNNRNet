import os
import utils
import platform

def get_params():
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
    params['batch_size']=120
    params['n_epochs']=3000
    params['dataset']=[]
    # dataset parameters

    if(platform.node()=="hc"):
        params['dataset'].append([])
        params['dataset'][0]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
        params['dataset'].append([])
        params['dataset'][1]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_teddy/"
        params['dataset'].append([])
        params['dataset'][2]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_cabinet/"
    if(platform.node()=="milletari-workstation"):
        params['dataset'].append([])
        params['dataset'][0]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
        params['dataset'].append([])
        params['dataset'][1]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_teddy/"
        params['dataset'].append([])
        params['dataset'][2]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_cabinet/"
        params['dataset'].append([])
        params['dataset'][3]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_coke/"
        params['dataset'].append([])
        params['dataset'][4]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_flowerbouquet/"
        params['dataset'].append([])
        params['dataset'][5]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_flowerbouquet_brownbackground/"
        params['dataset'].append([])
        params['dataset'][6]="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_dishes/"
    if(platform.node()=="std"):
        params['dataset'].append([])
        params['dataset'][0]="/home/cmp/projects/data/rgbd_dataset_freiburg3_large_cabinet/"
        params['dataset'].append([])
        params['dataset'][1]="/home/cmp/projects/data/rgbd_dataset_freiburg3_teddy/"
        params['dataset'].append([])
        params['dataset'][2]="/home/cmp/projects/data/rgbd_dataset_freiburg3_cabinet/"

    params['im_type']="depth"
    params['step_size']=[1,2,5,7,10,12,13,14,15,16,18,20,21,23,24,25]
    #params['step_size']=[10]
    params['size']=[160, 120] #[width,height]
    params['nc']=1 #number of dimensions
    params['multi']=10 #ground truth location differences will be multiplied with this number
    params['test_size']=0.20 #Test size
    params['val_size']=0.20 #Test size
    params['test_freq']=3 #Test frequency

    # c an Pooling parameters
    params['kern_mat']=[(5, 5), (5, 5)] #shape of kernel
    params['nkerns']= [40, 30,20,20] #number of kernel
    params['pool_mat']=  [(2, 2), (2, 2)] #shape of pooling

    # os
    wd=os.getcwd()
    params['wd']=wd
    params['models']=wd+"/models/"
    params['log_file']=wd+"/logs/log_"+utils.get_time()+".txt"
    params['model_name']=wd+"models/1_2.h5"

    params['check_mode']=1 #process checking

    return params

