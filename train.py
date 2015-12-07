import sys
import numpy as np
import argparse
import helper.data_loader as data_loader
from helper import config, utils, dt_utils
from models import model_provider

sys.setrecursionlimit(50000)

def train_model(params):
  rn_id=params["rn_id"]
  im_type=params["im_type"]
  batch_size =params["batch_size"]
  n_epochs =params["n_epochs"]

  datasets = data_loader.load_data(params)
  utils.start_log(datasets,params)

  X_train, y_train,overlaps_train = datasets[0]
  X_val, y_val,overlaps_val = datasets[1]
  X_test, y_test,overlaps_test = datasets[2]

  # compute number of minibatches for training, validation and testing
  n_train_batches = len(X_train)
  n_valid_batches = len(X_val)
  n_test_batches = len(X_test)
  n_train_batches /= batch_size
  n_valid_batches /= batch_size
  n_test_batches /= batch_size

  y_val_mean=np.mean(y_val)
  y_val_abs_mean=np.mean(np.abs(y_val))

  utils.log_write("Model build started",params)
  model=  model_provider.get_model(params)
  run_mode=params["run_mode"]
  utils.log_write("Model build ended",params)
  utils.log_write("Training started",params)
  best_validation_loss=np.inf
  epoch_counter = 0
  while (epoch_counter < n_epochs):
      epoch_counter = epoch_counter + 1
      print("Training model...")
      for minibatch_index in xrange(n_train_batches):
          iter = (epoch_counter - 1) * n_train_batches + minibatch_index
          if iter % 100 == 0:
              print 'training @ iter = ', iter

          Fx = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
          data_Fx = dt_utils.load_batch_images(params,"F", Fx)
          data_Sx = dt_utils.load_batch_images(params,"S", Fx)
          data_y = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
          loss =model.train_on_batch([data_Fx, data_Sx], data_y)
          if isinstance(loss,list):
             loss=loss[0]

          s='TRAIN--> epoch %i | minibatch %i/%i | error %f'%(epoch_counter, minibatch_index + 1, n_train_batches,  loss)
          utils.log_write(s,params)
          if(run_mode==1):
              break

      print("Validating model...")
      this_validation_loss = 0
      for i in xrange(n_valid_batches):
          Fx = X_val[i * batch_size: (i + 1) * batch_size]
          data_Fx = dt_utils.load_batch_images(params,"F", Fx)
          data_Sx = dt_utils.load_batch_images(params,"S", Fx)
          data_y = y_val[i * batch_size: (i + 1) * batch_size]
          loss= model.test_on_batch([data_Fx, data_Sx],data_y)
          if isinstance(loss,list):
             loss=loss[0]
          this_validation_loss +=loss
          if(run_mode==1):
              break
      this_validation_loss /=n_valid_batches

      s ='VAL--> epoch %i | error %f | data mean/abs %f/%f'%(epoch_counter, this_validation_loss,y_val_mean,y_val_abs_mean)
      utils.log_write(s,params)
      ext=params["model_file"]+params["model"]+"_"+str(epoch_counter % 10)+"_"+im_type+".hdf5"
      model.save_weights(ext, overwrite=True)
      if this_validation_loss < best_validation_loss:
          best_validation_loss = this_validation_loss
          ext=params["model_file"]+params["model"]+"_"+str(rn_id)+"_best_"+str(epoch_counter)+"_"+im_type+".hdf5"
          model.save_weights(ext, overwrite=True)

      #We are shuffling data at each epoch
      X_train,y_train=dt_utils.shuffle_in_unison_inplace(X_train,y_train)
      if(run_mode==1):
              break
  ext=params["model_file"]+params["model"]+"_regular_"+str(epoch_counter % 5)+"_"+im_type+".hdf5"
  model.save_weights(ext, overwrite=True)
  utils.log_write("Training ended",params)

if __name__ == "__main__":
  params= config.get_params()
  parser = argparse.ArgumentParser(description='Training the module')
  parser.add_argument('-rm','--run_mode',type=int, help='Training mode:0 full train, 1 system check, 2 simle train',
                      required=True)
  parser.add_argument('-m','--model',help='Model: kcnnr, dccnr, current('+params["model"]+')',default=params["model"])
  parser.add_argument('-i','--im_type',help='Image type: pre_depth, depth, gray, current('+params["im_type"]+')',
                      default=params["im_type"])

  args = vars(parser.parse_args())
  params["run_mode"]=args["run_mode"]
  params["model"]=args["model"]
  params["im_type"]=args["im_type"]
  params=config.update_params(params)
  train_model(params)