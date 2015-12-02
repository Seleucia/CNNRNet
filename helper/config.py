import os
import utils
import platform
import set_ds_list as sdl

def get_params():
   params={}
   params['check_mode']=1 #process checkY_testing
   params["rn_id"]="highreg" #running id
   params["notes"]="model l2 regularizers are increased" #running id
   params["model"]="mlpr"#kccnr,dccnr

   params['shufle_data']=0
   params['gray_mean']=114.33767967 #114.151092572
   params['depth_mean']=13797.3639853 #13746.3784954
   params['pre_depth_mean']=9505.32929609 #9515.98643977
   params['rgb_mean']=[138.28382874, 128.78469849 ,124.75618744] #[138.18440247,128.58282471 ,124.65019226]
   params['hha_depth_fc6_mean']=2
   params['batch_size']=240
   params['im_type']="hha_depth_fc6"
   params['step_size']=[1,5,7,10,12,14,15,17,19,21,23,25]

   #system settings
   wd=os.path.dirname(os.path.realpath(__file__))
   wd=os.path.dirname(wd)
   params['wd']=wd
   params['log_file']=wd+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+utils.get_time()+".txt"
   params["model_file"]=wd+"/cp/"


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
   params=sdl.set_list(params)

   if(platform.node()=="hc"):
       params["caffe"]="/home/coskun/sftpkg/caffe/python"
       params['batch_size']=10
       params["WITH_GPU"]=False

   if(platform.node()=="milletari-workstation"):
       params["caffe"]="/usr/local/caffe/python"
       params["WITH_GPU"]=True

   if(platform.node()=="cmp-comp"):
       params['batch_size']=60
       params["WITH_GPU"]=True
       params["caffe"]="/home/coskun/sftpkg/caffe/python"

   #params['step_size']=[10]
   params['size']=[160, 120] #[width,height]
   params['nc']=1 #number of dimensions
   params['multi']=10 #ground truth location differences will be multiplied with this number
   params['test_size']=0.20 #Test size
   params['val_size']=0.20 #Test size
   params['test_freq']=100 #Test frequency

   # c an Pooling parameters
   params['kern_mat']=[(5, 5), (5, 5)] #shape of kernel
   params['nkerns']= [40, 30,20,20] #number of kernel
   params['pool_mat']=  [(2, 2), (2, 2)] #shape of pooling


   #Feature extraction:
   params['orijinal_img']="rgb" #rgb,depth
   params['layer']="fc6" #rgb,depth
   # os

   return params

def update_params(params):
   params['log_file']=params["wd"]+"/logs/"+params["model"]+"_"+params["rn_id"]+"_"+utils.get_time()+".txt"

   if(params['check_mode']==1):
       params['step_size']=[10]
   if(params['check_mode']==2):
       params['step_size']=[1,10,25]

   if params['check_mode']==2:
      idx=range(0,len(params["dataset"]),3)
      for i in idx:
         params["dataset"][i]=-1

   return params
