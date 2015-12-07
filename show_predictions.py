import numpy as np

import helper.tum_dataset_loader as tdl
import pred.predict_location
from helper import config, model_saver, utils
from plot import plot_data

params= config.get_params()
is_test=1
cm_mul=10
id=17 #data will be loaded according to this id
params['step_size']=[10]
step=params['step_size'][0]

params['model_name']="dcnnr_4_pre_depth.hdf5"
params['model']="dcnnr"
params['im_type']="pre_depth"
prediction_name= params['model_name'].replace("/", " ").split()[-1] .replace(".h5","")
ext_raw_data=prediction_name+"raw_data"+"_"+str(step)+".pkl"
ext_err=prediction_name+"err"+"_"+str(step)+".pkl"
ext_y_pred=prediction_name+"y_pred"+"_"+str(step)+".pkl"
ext_yy_test=prediction_name+"yy_test"+"_"+str(step)+".pkl"
ext_yy_test_aug=prediction_name+"yy_test_aug"+"_"+str(step)+".pkl"
ext_y_test_gt=prediction_name+"y_test_gt"+"_"+str(step)+".pkl"
fig_name3d=params["wd"]+"/pred/img/"+ prediction_name + "_" + str(step) + "3d.png"
fig_namexyz= params["wd"]+"/pred/img/"+prediction_name + "_" + str(step) + "xyz.png"


#orijinal locations of camera
dsRawData=tdl.load_data(params["dataset"][id],params["im_type"],params["n_output"])
dir_list=dsRawData[0]
data_y_gt=dsRawData[1]
data_y_gt=data_y_gt*cm_mul*cm_mul

dsSplits=tdl.split_data(dir_list,id, data_y_gt,params["test_size"],params["val_size"])
tmp_X_train,y_train_delta_gt=dsSplits[0]
tmp_X_test,y_test_delta_gt=dsSplits[is_test]


#location differences with orijinal, step_size=1 setted this means only looking consequtive locations
X_test, y_delta_test_1, overlaps_test=tdl.prepare_data(1, tmp_X_test, y_test_delta_gt)


#location differences with step_size=step{10} data augmented with stplits
X_test_aug, y_delta_test_step, overlaps_test_step=tdl.prepare_data(step, tmp_X_test, y_test_delta_gt)


#location prediction over augmented data
y_delta_pred= pred.predict_location.predict(X_test_aug, params)
y_delta_pred=np.asarray(y_delta_pred)
y_delta_pred=y_delta_pred*cm_mul


#Print and plot error with different axes
err= y_delta_pred - np.squeeze(np.asarray(y_delta_test_step))
plot_data.plot_err(err, fig_namexyz)
mean_error=np.mean(np.abs(err))
print "Mean Error:"+str(mean_error)
print "Mean of gt values:"+str(np.mean(y_delta_test_step))
print "Mean of pred values:"+str(np.mean(y_delta_pred))

#camera location restored from augmented data
yy_test_step= utils.up_sample(overlaps_test_step, y_delta_test_step, step)
yy_test_step=yy_test_step.reshape(len(yy_test_step),3)
yy_test_step=np.vstack([y_test_delta_gt[0, :], yy_test_step])
yy_test_step=np.cumsum(yy_test_step, axis=0)


#camera location restored from predicted data
yy_pred= utils.up_sample(overlaps_test_step, y_delta_pred, step)
yy_pred=yy_pred.reshape(len(yy_pred),3)
yy_pred=np.vstack([y_test_delta_gt[0,:],yy_pred])
yy_pred=np.cumsum(yy_pred,axis=0)


#camera location restored from differences data
q=np.squeeze(np.asarray(y_delta_test_1))
w=np.squeeze(np.asarray(y_test_delta_gt))[0,:]
yy_test_1=np.vstack((w,q))
yy_test_1=np.cumsum(yy_test_1, axis=0)

#save generated data
model_saver.save_pred(ext_raw_data, data_y_gt,params)
model_saver.save_pred(ext_err, err,params)
model_saver.save_pred(ext_y_pred, yy_pred,params)
model_saver.save_pred(ext_yy_test, yy_test_1,params)
model_saver.save_pred(ext_yy_test_aug, yy_test_step,params)
model_saver.save_pred(ext_y_test_gt, y_test_delta_gt,params)


plot_data.plot_y([yy_test_step, np.array(yy_pred)], fig_name3d)


