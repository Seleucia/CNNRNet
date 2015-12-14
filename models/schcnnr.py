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

#########Model single channel######################
def build_model(params):
   l2=regularizers.l2(0.01)
   l2_out=regularizers.l2(0.001)

   model = Sequential()
   model.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='valid', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   model.add(PReLU())
   model.add(MaxPooling2D(pool_size=(2, 2)))

   model.add(Convolution2D(params["nkerns"][1], 3, 3,init='he_normal', W_regularizer=l2))
   model.add(PReLU())
   model.add(MaxPooling2D(pool_size=(2, 2)))

   model.add(Convolution2D(params["nkerns"][2], 2, 2,init='he_normal', W_regularizer=l2))
   model.add(PReLU())
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Flatten())

   model.add(Dense(400,init='he_normal', W_regularizer=l2))
   model.add(PReLU())

   model.add(Dense(400,init='he_normal', W_regularizer=l2_out))
   model.add(PReLU())

   model.add(Dense(400,init='he_normal'))
   model.add(Activation('linear'))

   model.add(Dense(params['n_output'],init='he_normal'))

   adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
   model.compile(loss='mean_squared_error', optimizer=adagrad)
   return model
