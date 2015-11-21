import  plot_data
import utils
import dataset_loader
import numpy
import predict_location
import model_saver

rn_id=1
size = [160, 120]
multi=10
dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
offset=0
max_difference=0.2
step_size=[2,5,10,13,15,18,20,23,25,27,30,32,38,35,40,45,50]
step_size=[18]
batch_size=30
model_name="models/1_2_model_numpy.npy"

prediction_name= model_name.replace("/", " ").split()[-1] .replace(".npy","")
ext_raw_data=prediction_name+"raw_data"+"_"+str(step_size[0])+".pkl"
ext_err=prediction_name+"err"+"_"+str(step_size[0])+".pkl"
ext_y_pred=prediction_name+"y_pred"+"_"+str(step_size[0])+".pkl"
ext_yy_test=prediction_name+"yy_test"+"_"+str(step_size[0])+".pkl"
ext_yy_test_aug=prediction_name+"yy_test_aug"+"_"+str(step_size[0])+".pkl"
ext_y_test_gt=prediction_name+"y_test_gt"+"_"+str(step_size[0])+".pkl"
fig_name3d= prediction_name + "_" + str(step_size[0]) + "3d.png"
fig_namexyz= prediction_name + "_" + str(step_size[0]) + "xyz.png"


#location differences data augmented with stplits
datasets_aug= dataset_loader.load_tum_dataV2(dataset, rn_id, multi,step_size)
(X_train_aug, y_train_aug,overlaps_train_aug)=datasets_aug[0]
(X_val_aug, y_val_aug,overlaps_val_aug)=datasets_aug[1]
(X_test_aug, y_test_aug,overlaps_test_aug)=datasets_aug[2]

#location prediction over augmented data
y_pred=predict_location.predict(X_test_aug,model_name,batch_size,size)
n_test_batches = len(X_test_aug)
n_test_batches /= batch_size



#Pring and plot error with different axes
pred_y_test_aug=y_test_aug[0:n_test_batches*batch_size]
err=y_pred - pred_y_test_aug
plot_data.plot_err(err,fig_namexyz)
mean_error=numpy.mean(numpy.abs(err))
print "Mean Error:"+str(mean_error)
print "Mean of gt values:"+str(numpy.mean(pred_y_test_aug))
print "Mean of pred values:"+str(numpy.mean(y_pred))


#location differences with orijinal, step_size=1 setted this means only looking consequtive locations
datasets= dataset_loader.load_tum_dataV2(dataset, rn_id, multi, [1])
(X_train, y_train,overlaps_train)=datasets[0]
(X_val, y_val,overlaps_val)=datasets[1]
(X_test, y_test,overlaps_test)=datasets[2]


#orijinal locations of camera
partitions=dataset_loader.load_splits(dataset,offset,max_difference)
(X_train_gt, y_train_gt)=partitions[0]
(X_val_gt, y_val_gt)=partitions[1]
(X_test_gt, y_test_gt)=partitions[2]
(data_x, data_y)=partitions[3]


#camera location restored from augmented data
yy_test_aug=utils.up_sample(overlaps_test_aug,y_test_aug,step_size[0])
yy_test_aug=numpy.vstack([y_test_gt[0,:],yy_test_aug])
yy_test_aug=numpy.cumsum(yy_test_aug,axis=0)


#camera location restored from predicted data
pred_overlaps_test_aug=overlaps_test_aug[0:n_test_batches*batch_size]
yy_pred=utils.up_sample(pred_overlaps_test_aug,y_pred,step_size[0])
yy_pred=numpy.vstack([y_test_gt[0,:],yy_pred])
yy_pred=numpy.cumsum(yy_pred,axis=0)




#camera location restored from differences data
yy_test=numpy.vstack([y_test_gt[0,:],y_test])
yy_test=numpy.cumsum(yy_test,axis=0)



model_saver.save_pred(ext_raw_data,data_y)
model_saver.save_pred(ext_err,err)
model_saver.save_pred(ext_y_pred,yy_pred)
model_saver.save_pred(ext_yy_test,yy_test)
model_saver.save_pred(ext_yy_test_aug,yy_test_aug)
model_saver.save_pred(ext_y_test_gt,y_test_gt)


plot_data.plot_y([yy_test,yy_pred], fig_name3d)


print("ok")