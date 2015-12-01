from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Merge
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras import regularizers
import sys

sys.setrecursionlimit(50000)

#########Model without droupout######################
def build_model(params):
   l2=regularizers.l2(0.001)
   l2_out=regularizers.l2(0.0001)

   #########Left Stream######################
   lmodel = Sequential()
   lmodel.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='full', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))
   lmodel.add(BatchNormalization())


   lmodel.add(Convolution2D(params["nkerns"][1], 3, 3,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))
   lmodel.add(BatchNormalization())

   lmodel.add(Convolution2D(params["nkerns"][2], 2, 2,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(MaxPooling2D(pool_size=(2, 2)))
   lmodel.add(BatchNormalization())


   lmodel.add(Flatten())
   lmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(BatchNormalization())

   lmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   lmodel.add(PReLU())
   lmodel.add(BatchNormalization())

   #########Right Stream######################
   rmodel = Sequential()
   rmodel.add(Convolution2D(params["nkerns"][0], 3, 3, border_mode='full', input_shape=(params["nc"], params["size"][1], params["size"][0]),init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))
   rmodel.add(BatchNormalization())

   rmodel.add(Convolution2D(params["nkerns"][1], 3, 3,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))
   rmodel.add(BatchNormalization())

   rmodel.add(Convolution2D(params["nkerns"][3], 2, 2,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(MaxPooling2D(pool_size=(2, 2)))
   rmodel.add(BatchNormalization())

   rmodel.add(Flatten())
   rmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(BatchNormalization())

   rmodel.add(Dense(200,init='he_normal', W_regularizer=l2))
   rmodel.add(PReLU())
   rmodel.add(BatchNormalization())

   #########Merge Stream######################
   model = Sequential()
   model.add(Merge([lmodel, rmodel], mode='mul'))
   model.add(Dense(400,init='he_normal', W_regularizer=l2_out))
   model.add(PReLU())

   model.add(Dense(400,init='he_normal'))
   model.add(Activation('linear'))

   model.add(Dense(3,init='he_normal'))

   sgd = SGD(lr=params['initial_learning_rate'], decay=params['learning_rate_decay'], momentum=params['momentum'], nesterov=True)
   adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
   model.compile(loss='mean_squared_error', optimizer=adagrad)
   model.compile(loss='mean_squared_error', optimizer=adagrad)
   return model