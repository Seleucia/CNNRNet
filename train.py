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
  utils.log_write("Number of parameters: %s"%(model.count_params()),params)
  run_mode=params["run_mode"]
  utils.log_write("Model build ended",params)
  utils.log_write("Training started",params)
  best_validation_loss=np.inf
  epoch_counter = 0
  n_patch=params["n_patch"]
  n_repeat=params["n_repeat"]#instead of extracting many batches for each epoch, we are repeating epoch since we are ensuring that output changes for each patch
  while (epoch_counter < n_epochs):
      epoch_counter = epoch_counter + 1
      print("Training model...")
      for index in xrange(n_train_batches*n_repeat):
          minibatch_index=index%n_train_batches
          #We are shuffling data at each batch, we are shufling here because we already finished one epoch just repeating for the extract different batch
          if(index>0 and minibatch_index==0):#we are checking weather we finish all dataset
             ext=params["model_file"]+params["model"]+"_"+im_type+"_m_"+str(index%5)+".hdf5"
             model.save_weights(ext, overwrite=True)
             X_train,y_train=dt_utils.shuffle_in_unison_inplace(X_train,y_train)

          iter = (epoch_counter - 1) * n_train_batches + index
          if iter % 100 == 0:
              print 'training @ iter = ', iter
          batch_loss=0
          Fx = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
          data_y = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
          for patch_index in xrange(n_patch):
             patch_loc=utils.get_patch_loc(params)
             argu= [(params,"F", Fx,patch_loc),(params,"S", Fx,patch_loc)]
             results = dt_utils.asyn_load_batch_images(argu)
             data_Fx = results[0]
             data_Sx = results[1]
             loss =model.train_on_batch([data_Fx, data_Sx], data_y)
             if isinstance(loss,list):
                batch_loss+=loss[0]
             else:
                batch_loss+=loss

          batch_loss/=n_patch
          s='TRAIN--> epoch %i | batch_index %i/%i | error %f'%(epoch_counter, index + 1, n_train_batches*n_repeat,  batch_loss)
          utils.log_write(s,params)
          if(run_mode==1):
              break
      #we are shufling for to be sure
      X_train,y_train=dt_utils.shuffle_in_unison_inplace(X_train,y_train)
      ext=params["model_file"]+params["model"]+"_"+im_type+"_e_"+str(epoch_counter % 10)+".hdf5"
      model.save_weights(ext, overwrite=True)

      if params['validate']==0:
         print("Validation skipped...")
         if(run_mode==1):
              break
         continue
      print("Validating model...")
      this_validation_loss = 0
      for index in xrange(n_valid_batches*n_repeat):
         i = index%n_valid_batches
         epoch_loss=0
         Fx = X_val[i * batch_size: (i + 1) * batch_size]
         data_y = y_val[i * batch_size: (i + 1) * batch_size]
         for patch_index in xrange(n_patch):
            patch_loc=utils.get_patch_loc(params)
            argu= [(params,"F", Fx,patch_loc),(params,"S", Fx,patch_loc)]
            results = dt_utils.asyn_load_batch_images(argu)
            data_Fx = results[0]
            data_Sx = results[1]
            loss= model.test_on_batch([data_Fx, data_Sx],data_y)
            if isinstance(loss,list):
                epoch_loss+=loss[0]
            else:
               epoch_loss+=loss
         epoch_loss/=n_patch
         this_validation_loss +=epoch_loss
         if(run_mode==1):
              break
      this_validation_loss /= (n_valid_batches*n_repeat)
      s ='VAL--> epoch %i | error %f | data mean/abs %f/%f'%(epoch_counter, this_validation_loss,y_val_mean,y_val_abs_mean)
      utils.log_write(s,params)
      if this_validation_loss < best_validation_loss:
          best_validation_loss = this_validation_loss
          ext=params["model_file"]+params["model"]+"_"+im_type+"_"+"_best_"+str(rn_id)+"_"+str(epoch_counter)+".hdf5"
          model.save_weights(ext, overwrite=True)
      if(run_mode==1):
              break
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
  params["is_exit"]=0
  params=config.update_params(params)
  try:
     train_model(params)
  except KeyboardInterrupt:
     utils.log_write("Exiting program",params)
     params["is_exit"]=1
  except Exception, e:
        utils.log_write('got exception: %r, terminating the pool' % (e,),params)
        params["is_exit"]=1