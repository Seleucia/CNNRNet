import helper.data_loader as data_loader
import helper.dt_utils as dt_utils
import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Merge
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras import regularizers
import sys
from helper import config, utils

sys.setrecursionlimit(50000)

def build_model(params):
    lmodel = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    lmodel.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='full', input_shape=(params["nc"], params["size"][1], params["size"][0])))
    lmodel.add(PReLU())
    lmodel.add(MaxPooling2D(pool_size=(2, 2)))
    lmodel.add(Convolution2D(params["nkerns"][1], 3, 3))
    lmodel.add(PReLU())
    lmodel.add(MaxPooling2D(pool_size=(2, 2)))
    lmodel.add(Convolution2D(params["nkerns"][2], 3, 3))
    lmodel.add(PReLU())
    lmodel.add(MaxPooling2D(pool_size=(2, 2)))


    lmodel.add(Flatten())
    lmodel.add(Dense(200))
    lmodel.add(PReLU())
    lmodel.add(Dense(200))
    lmodel.add(PReLU())

    rmodel = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    rmodel.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='full', input_shape=(params["nc"], params["size"][1], params["size"][0])))
    rmodel.add(PReLU())
    rmodel.add(MaxPooling2D(pool_size=(2, 2)))
    rmodel.add(Convolution2D(params["nkerns"][1], 3, 3))
    rmodel.add(PReLU())
    rmodel.add(MaxPooling2D(pool_size=(2, 2)))
    rmodel.add(Convolution2D(params["nkerns"][3], 3, 3))
    rmodel.add(PReLU())
    rmodel.add(MaxPooling2D(pool_size=(2, 2)))

    rmodel.add(Flatten())
    rmodel.add(Dense(200))
    rmodel.add(PReLU())
    rmodel.add(Dense(200))
    rmodel.add(PReLU())

    model = Sequential()
    model.add(Merge([lmodel, rmodel], mode='mul'))
    model.add(Dense(400))
    model.add(PReLU())
    model.add(Dense(400))
    model.add(Activation('linear'))

    model.add(Dense(3))

    sgd = SGD(lr=params['initial_learning_rate'], decay=params['learning_rate_decay'], momentum=params['momentum'], nesterov=True)
    adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
    reg2=regularizers.l2(0.01)
    model.regularizers.append(reg2)
    model.compile(loss='mean_squared_error', optimizer=adagrad)
    return model

def train_model(params):
    rn_id=params["rn_id"]
    im_type=params["im_type"]
    batch_size =params["batch_size"]
    n_epochs =params["n_epochs"]
    nc =params["nc"]  # number of channcels
    size =params["size"]  # size = [480,640] orijinal size,[height,width]

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

    # n_train_batches = 3
    # n_valid_batches = 1
    # n_test_batches = 1

    y_val_mean=np.mean(y_val)
    y_val_abs_mean=np.mean(np.abs(y_val))
    y_test_mean=np.mean(y_test)
    y_test_abs_mean=np.mean(np.abs(y_test))

    utils.log_write("Model build started",params)
    model=  build_model(params)
    check_mode=params["check_mode"]
    utils.log_write("Model build ended",params)
    utils.log_write("Training started",params)
    done_looping = False
    epoch_counter = 0
    best_validation_loss=np.inf
    test_counter=0
    while (epoch_counter < n_epochs) and (not done_looping):
        epoch_counter = epoch_counter + 1
        print("Training model...")
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch_counter - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print 'training @ iter = ', iter

            Fx = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            data_Fx = dt_utils.load_batch_images(size, nc, "F", Fx,im_type)
            data_Sx = dt_utils.load_batch_images(size, nc, "S", Fx,im_type)
            data_y = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            loss =model.train_on_batch([data_Fx, data_Sx], data_y)
            s='TRAIN--> epoch %i | minibatch %i/%i | error %f'%(epoch_counter, minibatch_index + 1, n_train_batches,  loss)
            utils.log_write(s,params)
            if(check_mode==1):
                break

        print("Validating model...")
        this_validation_loss = 0
        for i in xrange(n_valid_batches):
            Fx = X_val[i * batch_size: (i + 1) * batch_size]
            data_Fx = dt_utils.load_batch_images(size, nc, "F", Fx,im_type)
            data_Sx = dt_utils.load_batch_images(size, nc, "S", Fx,im_type)
            data_y = y_val[i * batch_size: (i + 1) * batch_size]
            this_validation_loss += model.test_on_batch([data_Fx, data_Sx],data_y)
            if(check_mode==1):
                break
        this_validation_loss /=n_valid_batches

        s ='VAL--> epoch %i | error %f | data mean/abs %f/%f'%(epoch_counter, this_validation_loss,y_val_mean,y_val_abs_mean)
        utils.log_write(s,params)
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            # ext=params["models"]+str(rn_id)+"_"+str(epoch_counter % 3)+"_"+im_type+".pkl"
            # pickle.dump(model, open(ext, 'wb'))
            ext=params["models"]+str(rn_id)+"_"+str(epoch_counter % 3)+"_"+im_type+".hdf5"
            model.save_weights(ext, overwrite=True)
            test_counter+=1
            if(test_counter%params["test_freq"]==1):
                test_losses = 0
                for i in xrange(n_test_batches):
                    Fx = X_test[i * batch_size: (i + 1) * batch_size]
                    data_Fx = dt_utils.load_batch_images(size, nc, "F", Fx,im_type)
                    data_Sx = dt_utils.load_batch_images(size, nc, "S", Fx,im_type)
                    data_y = y_test[i * batch_size: (i + 1) * batch_size]
                    test_losses +=  model.test_on_batch([data_Fx, data_Sx],data_y)
                    if(check_mode==1):
                        break
                test_losses/=n_test_batches
                s ='TEST--> epoch %i | error %f | data mean/abs %f / %f' %(epoch_counter, test_losses,y_test_mean,y_test_abs_mean)
                utils.log_write(s,params)
        else:
            if(epoch_counter % 5==0):
                # ext=params["models"]+str(rn_id)+"_regular_"+str(epoch_counter % 5)+"_"+im_type+".pkl"
                # pickle.dump(model, open(ext, 'wb'))
                ext=params["models"]+str(rn_id)+"_regular_"+str(epoch_counter % 5)+"_"+im_type+".hdf5"
                model.save_weights(ext, overwrite=True)

        if(check_mode==1):
                break
    # ext=params["models"]+"last_"+str(rn_id)+utils.get_time()+"_"+im_type+".pkl"
    # pickle.dump(model, open(ext, 'wb'))
    ext=params["models"]+str(rn_id)+"_regular_"+str(epoch_counter % 5)+"_"+im_type+".hdf5"
    model.save_weights(ext, overwrite=True)
    utils.log_write("Training ended",params)

if __name__ == "__main__":
    params= config.get_params()
    train_model(params)