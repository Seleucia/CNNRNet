import os
import utils
import platform

def get_params():
    params={}
    params['check_mode']=1 #process checking
    params["rn_id"]="val_drop_wight" #running id
    params["notes"]="Batch normalaziton removed, dropout and l2 regularize added, weight init setting norm" #running id

    params['shufle_data']=1
    params['gray_mean']=130.668652173 #130.507121448
    params['depth_mean']=13797.3639853 #13746.3784954
    params['pre_depth_mean']=9505.32929609 #9515.98643977
    params['rgb_mean']=[138.28382874, 128.78469849 ,124.75618744] #[138.18440247,128.58282471 ,124.65019226]
    params['batch_size']=60


    # early-stopping parameters
    params['patience']= 10000  # look as this many examples regardless
    params['patience_increase']=2  # wait this much longer when a new best is
    params['improvement_threshold']=0.995  # a relative improvement of this much is

    # learning parameters
    params['momentum']=0.9    # the params for momentum
    params['initial_learning_rate']=0.001
    params['learning_rate_decay']= 0.998
    params['squared_filter_length_limit']=15.0
    params['n_epochs']=3000

    # dataset parameters
    params['dataset']=[]

    if(platform.node()=="hc"):
        params["caffe"]="/home/coskun/sftpkg/caffe/python"
        params['batch_size']=10
        params["WITH_GPU"]=False
        params['dataset'].append([])
        params['dataset'][0]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/","TUM"]
        params['dataset'].append([])
        params['dataset'][1]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_teddy/","TUM"]
        params['dataset'].append([])
        params['dataset'][2]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_cabinet/","TUM"]
        params['dataset'].append([])
        params['dataset'][3]=["/home/coskun/PycharmProjects/data/living_room_traj0_frei_png/","ICL"]
    if(platform.node()=="milletari-workstation"):
        params["caffe"]="/usr/local/caffe/python"
        params["WITH_GPU"]=True
        params['dataset'].append([])
        params['dataset'][0]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/","TUM"]
        params['dataset'].append([])
        params['dataset'][1]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_teddy/","TUM"]
        params['dataset'].append([])
        params['dataset'][2]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_cabinet/","TUM"]
        params['dataset'].append([])
        params['dataset'][3]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_coke/","TUM"]
        params['dataset'].append([])
        params['dataset'][4]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_flowerbouquet/","TUM"]
        params['dataset'].append([])
        params['dataset'][5]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_flowerbouquet_brownbackground/","TUM"]
        params['dataset'].append([])
        params['dataset'][6]=["/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg2_dishes/","TUM"]
    if(platform.node()=="cmp-comp"):
        params["WITH_GPU"]=False
        params["caffe"]="/home/coskun/sftpkg/caffe/python"
        params['dataset'].append([])
        params['dataset'][0]=["/home/cmp/projects/data/rgbd_dataset_freiburg3_large_cabinet/","TUM"]
        params['dataset'].append([])
        params['dataset'][1]=["/home/cmp/projects/data/rgbd_dataset_freiburg3_teddy/","TUM"]
        params['dataset'].append([])
        params['dataset'][2]=["/home/cmp/projects/data/rgbd_dataset_freiburg3_cabinet/","TUM"]
        params['dataset'].append([])
        params['dataset'][3]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_coke/","TUM"]
        params['dataset'].append([])
        params['dataset'][4]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_flowerbouquet/","TUM"]
        params['dataset'].append([])
        params['dataset'][5]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_flowerbouquet_brownbackground/","TUM"]
        params['dataset'].append([])
        params['dataset'][6]=["/home/cmp/projects/data/rgbd_dataset_freiburg2_dishes/","TUM"]

    params['im_type']="depth"
    params['step_size']=[1,2,5,7,10,12,13,14,15,16,18,20,21,23,24,25]
    #params['step_size']=[10]
    params['size']=[160, 120] #[width,height]
    params['nc']=1 #number of dimensions
    params['multi']=10 #ground truth location differences will be multiplied with this number
    params['test_size']=0.20 #Test size
    params['val_size']=0.20 #Test size
    params['test_freq']=10 #Test frequency

    # c an Pooling parameters
    params['kern_mat']=[(5, 5), (5, 5)] #shape of kernel
    params['nkerns']= [40, 30,20,20] #number of kernel
    params['pool_mat']=  [(2, 2), (2, 2)] #shape of pooling


    #Feature extraction:
    params['orijinal_img']="depth" #rgb,depth
    params['layer']="fc7" #rgb,depth
    # os

    wd=os.path.dirname(os.path.realpath(__file__))
    wd=os.path.dirname(wd)
    params['wd']=wd
    params['models']=wd+"/models/"
    params['log_file']=wd+"/logs/log_"+params["rn_id"]+"_"+utils.get_time()+".txt"
    params['model_name']=wd+"models/1_2.h5"

    if(params['check_mode']==1):
        params['step_size']=[10,15]

    return params

