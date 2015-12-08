from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Merge
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras import regularizers
import sys

sys.setrecursionlimit(50000)

#########Model without droupout######################
def build_model(params):
   l2=regularizers.l2(0.01)
   l2_out=regularizers.l2(0.001)
   nkern=16

   #########Left Stream######################
   lmodel = Sequential()
   lmodel.add(Convolution2D(nkern, 7, 7,subsample=params['stride_mat'], border_mode='valid', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))

   lmodel.add(Flatten())
   lmodel.add(Dense(256,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())

   lmodel.add(Dense(256,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())

   #########Right Stream######################
   rmodel = Sequential()
   rmodel.add(Convolution2D(nkern, 7, 7,subsample=params['stride_mat'], border_mode='valid', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))

   rmodel.add(Flatten())
   rmodel.add(Dense(256,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())

   rmodel.add(Dense(256,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   #########Merge Stream######################
   model = Sequential()
   model.add(Merge([lmodel, rmodel], mode='mul'))
   model.add(Dense(256,init='he_normal', W_regularizer=l2_out))
   model.add(PReLU())

   model.add(Dense(256,init='he_normal'))
   model.add(Activation('linear'))

   model.add(Dense(params['n_output'],init='he_normal'))

   adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
   model.compile(loss='mean_squared_error', optimizer=adagrad)
   return model
