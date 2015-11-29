import helper.tum_dataset_loader as tdl
from helper import config, model_saver, utils
from plot import plot_data
import numpy as np

params= config.get_params()

id=4 #data will be loaded according to this id
params['shufle_data']=0
params['im_type']="depth"
params['step_size']=[10]
step=params['step_size'][0]

#orijinal locations of camera
dsRawData=tdl.load_data(params["dataset"][id],params["im_type"])
dir_list=dsRawData[0]
data_y_gt=dsRawData[1]

dsSplits=tdl.split_data(dir_list,id, data_y_gt,params["test_size"],params["val_size"])
tmp_X_train,y_train_delta_gt=dsSplits[0]
tmp_X_test,y_test_delta_gt=dsSplits[1]
tmp_X_val,y_val_delta_gt=dsSplits[2]

plot_data.plot_orijinal_y(y_train_delta_gt,y_test_delta_gt,y_val_delta_gt)


print("ok")