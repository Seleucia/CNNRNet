from keras.layers.core import Merge
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adagrad
import dataset_loader
import utils
import numpy as np
import os


def build_model(params):

    lmodel = Sequential()

    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    lmodel.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='full', input_shape=(params["nc"], params["size"][1], params["size"][0])))
    lmodel.add(Activation('relu'))
    lmodel.add(Convolution2D(params["nkerns"][1], 3, 3))
    lmodel.add(Activation('relu'))
    lmodel.add(MaxPooling2D(pool_size=(2, 2)))
    lmodel.add(Dropout(0.25))

    lmodel.add(Flatten())
    # Note: Keras does automatic shape inference.
    lmodel.add(Dense(256))
    lmodel.add(Activation('relu'))
    lmodel.add(Dropout(0.5))

    lmodel.add(Dense(256))
    lmodel.add(Activation('relu'))
    lmodel.add(Dropout(0.5))

    rmodel = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    rmodel.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='full', input_shape=(params["nc"], params["size"][1], params["size"][0])))
    rmodel.add(Activation('relu'))
    rmodel.add(Convolution2D(params["nkerns"][1], 3, 3))
    rmodel.add(Activation('relu'))
    rmodel.add(MaxPooling2D(pool_size=(2, 2)))
    rmodel.add(Dropout(0.25))

    rmodel.add(Flatten())
    # Note: Keras does automatic shape inference.
    rmodel.add(Dense(256))
    rmodel.add(Activation('relu'))
    rmodel.add(Dropout(0.5))

    rmodel.add(Dense(256))
    rmodel.add(Activation('relu'))
    rmodel.add(Dropout(0.5))

    model = Sequential()
    model.add(Merge([lmodel, rmodel], mode='mul'))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('relu'))
    sgd = SGD(lr=params['initial_learning_rate'], decay=params['learning_rate_decay'], momentum=params['momentum'], nesterov=True)
    adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
    model.compile(loss='mean_squared_error', optimizer=adagrad)
    return model


def train_model(params):
    rn_id=params["rn_id"]
    im_type=params["im_type"]
    batch_size =params["batch_size"]
    n_epochs =params["n_epochs"]
    nc =params["nc"]  # number of channcels
    size =params["size"]  # size = [480,640] orijinal size,[height,width]

    datasets = dataset_loader.load_tum_dataV2(params)
    X_train, y_train,overlaps_train = datasets[0]
    X_val, y_val,overlaps_val = datasets[1]
    X_test, y_test,overlaps_test = datasets[2]
    print("Database loaded")
    # compute number of minibatches for training, validation and testing
    n_train_batches = len(X_train)
    n_valid_batches = len(X_val)
    n_test_batches = len(X_test)
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    model=  build_model(params)
    print("Model builded")
    done_looping = False
    epoch_counter = 0
    best_validation_loss=np.inf
    print("Mean of training data:%f, abs mean",np.mean(y_train),np.mean(np.abs(y_train)))
    print("Mean of val data:%f, abs mean: ",np.mean(y_val),np.mean(np.abs(y_val)))
    print("Mean of test data:%f, abs mean",np.mean(y_test),np.mean(np.abs(y_test)))
    while (epoch_counter < n_epochs) and (not done_looping):
        epoch_counter = epoch_counter + 1
        print("Training model...")
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch_counter - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print 'training @ iter = ', iter

            Fx = X_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx,im_type)
            data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx,im_type)
            data_y = y_train[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            loss =model.train_on_batch([data_Fx, data_Sx], data_y)
            s='TRAIN--> epoch %i, minibatch %i/%i, training cost %f ' % (epoch_counter, minibatch_index + 1, n_train_batches,  loss)
            utils.log_write(s)

        print("Validating model...")
        this_validation_loss = 0
        val_mean = 0
        val_abs_mean=0
        for i in xrange(n_valid_batches):
                    Fx = X_val[i * batch_size: (i + 1) * batch_size]
                    data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx,im_type)
                    data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx,im_type)
                    data_y = y_val[i * batch_size: (i + 1) * batch_size]
                    val_mean+=np.mean(data_y)
                    val_abs_mean+=np.mean(np.abs(data_y))
                    test_losses +=  model.test_on_batch([data_Fx, data_Sx],data_y)
                    this_validation_loss += model.test_on_batch([data_Fx, data_Sx],data_y)
        this_validation_loss /=n_valid_batches
        val_abs_mean/=n_test_batches
        val_mean/=n_test_batches
        s ='VAL--> epoch %i, validation error %f val data mean %f prediction mean %f prediction abs mean %%' %(epoch_counter, this_validation_loss,val_mean,val_abs_mean)
        utils.log_write(s)
        if this_validation_loss < best_validation_loss:
            best_validation_loss = this_validation_loss
            test_losses = 0
            test_mean = 0
            test_abs_mean=0
            for i in xrange(n_test_batches):
                Fx = X_test[i * batch_size: (i + 1) * batch_size]
                data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx,im_type)
                data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx,im_type)
                data_y = y_test[i * batch_size: (i + 1) * batch_size]
                test_mean+=np.mean(data_y)
                test_abs_mean+=np.mean(np.abs(data_y))
                test_losses +=  model.test_on_batch([data_Fx, data_Sx],data_y)
            test_losses/=n_test_batches
            test_abs_mean/=n_test_batches
            test_mean/=n_test_batches
            ext=params["models"]+str(rn_id)+"_"+str(epoch_counter % 3)+".h5"
            model.save_weights(ext, overwrite=True)
            s ='TEST--> epoch %i, test error %f test data mean %f prediction mean %f prediction abs mean %%' %(epoch_counter, test_losses,test_mean,test_abs_mean)
            utils.log_write(s)



if __name__ == "__main__":

    params={}
    params["rn_id"]=1 #running id
    # early-stopping parameters
    params['patience']= 10000  # look as this many examples regardless
    params['patience_increase']=2  # wait this much longer when a new best is
    params['improvement_threshold']=0.995  # a relative improvement of this much is

    # learning parameters
    params['lambda_1']= 0.01  # regulizer param
    params['lambda_2']=0.01  # regulizer param
    params['momentum']=0.9    # the params for momentum
    params['initial_learning_rate']=0.001
    params['learning_rate_decay']= 0.998
    params['squared_filter_length_limit']=15.0
    params['batch_size']=30
    params['n_epochs']=3000

    # dataset parameters
    #params['dataset']="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
    params['dataset']="/home/cmp/projects/data/rgbd_dataset_freiburg3_large_cabinet/" #test computer
    params['im_type']="gray"
    params['step_size']=[1,2,5,7,10,12,13,14,15,16,18,20,21,23,24,25]
    params['size']=[160, 120] #[width,height]
    params['nc']=1 #number of dimensions
    params['multi']=10 #ground truth location differences will be multiplied with this number
    params['test_size']=0.20 #Test size
    params['val_size']=0.20 #Test size

    # c an Pooling parameters
    params['kern_mat']=[(5, 5), (5, 5)] #shape of kernel
    params['nkerns']= [30, 40] #number of kernel
    params['pool_mat']=  [(2, 2), (2, 2)] #shape of pooling

    # os
    wd=os.getcwd()
    params['wd']=wd
    params['models']=wd+"/models/"
    params['logs']=wd+"/logs/"

    train_model(params)