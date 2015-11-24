import dataset_loader
import numpy

import kcnnr
import plot_data
import predict_location
from helper import config, model_saver, utils

params= config.get_params("home")
params['step_size']=[10]
step=params['step_size'][0]

prediction_name= params['model_name'].replace("/", " ").split()[-1] .replace(".npy","")
ext_raw_data=prediction_name+"raw_data"+"_"+str(step)+".pkl"
ext_err=prediction_name+"err"+"_"+str(step)+".pkl"
ext_y_pred=prediction_name+"y_pred"+"_"+str(step)+".pkl"
ext_yy_test=prediction_name+"yy_test"+"_"+str(step)+".pkl"
ext_yy_test_aug=prediction_name+"yy_test_aug"+"_"+str(step)+".pkl"
ext_y_test_gt=prediction_name+"y_test_gt"+"_"+str(step)+".pkl"
fig_name3d= prediction_name + "_" + str(step) + "3d.png"
fig_namexyz= prediction_name + "_" + str(step) + "xyz.png"


#orijinal locations of camera
partitions=dataset_loader.load_splits(params)
(X_test_gt, y_test_gt)=partitions[2]
(data_x, data_y)=partitions[3]

kcnnr.build_model(params)
#location differences with orijinal, step_size=1 setted this means only looking consequtive locations
X_test,y_delta_test,overlaps_test=dataset_loader.prepare_data(1,X_test_gt,y_test_gt)


#location differences data augmented with stplits
X_test_aug, y_delta_test_aug, overlaps_test_aug=dataset_loader.prepare_data(step,X_test_gt,y_test_gt)

#location prediction over augmented data
y_delta_pred= predict_location.predict(X_test_aug, params)
n_test_batches = len(X_test_aug)
n_test_batches /= params['batch_size']



#Print and plot error with different axes
pred_y_delta_test_aug= y_delta_test_aug[0:n_test_batches * params['batch_size']]
err= y_delta_pred - pred_y_delta_test_aug
plot_data.plot_err(err,fig_namexyz)
mean_error=numpy.mean(numpy.abs(err))
print "Mean Error:"+str(mean_error)
print "Mean of gt values:"+str(numpy.mean(pred_y_delta_test_aug))
print "Mean of pred values:"+str(numpy.mean(y_delta_pred))

#camera location restored from augmented data
yy_test_aug= utils.up_sample(overlaps_test_aug, y_delta_test_aug, step)
yy_test_aug=numpy.vstack([y_test_gt[0,:],yy_test_aug])
yy_test_aug=numpy.cumsum(yy_test_aug,axis=0)


#camera location restored from predicted data
pred_overlaps_test_aug=overlaps_test_aug[0:n_test_batches*params['batch_size']]
yy_pred= utils.up_sample(pred_overlaps_test_aug, y_delta_pred, step)
yy_pred=numpy.vstack([y_test_gt[0,:],yy_pred])
yy_pred=numpy.cumsum(yy_pred,axis=0)


#camera location restored from differences data
yy_test=numpy.vstack([y_test_gt[0,:], y_delta_test])
yy_test=numpy.cumsum(yy_test,axis=0)

#save generated data
model_saver.save_pred(ext_raw_data, data_y)
model_saver.save_pred(ext_err, err)
model_saver.save_pred(ext_y_pred, yy_pred)
model_saver.save_pred(ext_yy_test, yy_test)
model_saver.save_pred(ext_yy_test_aug, yy_test_aug)
model_saver.save_pred(ext_y_test_gt, y_test_gt)


plot_data.plot_y([y_test_gt,yy_test], fig_name3d)


print("ok")