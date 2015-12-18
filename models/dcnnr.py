from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Merge
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras import regularizers
import sys

sys.setrecursionlimit(50000)

#########Model with droupout######################

def build_model(params):
   l2=regularizers.l2(0.0005)
   l2_out=regularizers.l2(0.00005)

   #########Left Stream######################
   lmodel = Sequential()
   lmodel.add(Convolution2D(params["nkerns"][0], 7, 7,subsample=params['stride_mat'], border_mode='valid', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))

   lmodel.add(Convolution2D(params["nkerns"][1], 3, 3,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))

   lmodel.add(Convolution2D(params["nkerns"][2], 2, 2,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))
   lmodel.add(Flatten())
   lmodel.add(Dropout(0.5))


   lmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())


   lmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())

   #########Right Stream######################
   rmodel = Sequential()
   rmodel.add(Convolution2D(params["nkerns"][0], 7, 7,subsample=params['stride_mat'], border_mode='valid', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))

   rmodel.add(Convolution2D(params["nkerns"][1], 3, 3,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))

   rmodel.add(Convolution2D(params["nkerns"][3], 2, 2,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))
   rmodel.add(Flatten())
   rmodel.add(Dropout(0.5))

   rmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())


   rmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())

   #########Merge Stream######################
   model = Sequential()
   model.add(Merge([lmodel, rmodel], mode='mul'))
   model.add(Dense(200,init='he_normal', W_regularizer=l2_out))
   model.add(PReLU())

   model.add(Dense(200,init='he_normal'))
   model.add(Activation('linear'))

   model.add(Dense(params['n_output'],init='he_normal'))

   adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
   model.compile(loss='mean_squared_error', optimizer=adagrad)
   return model