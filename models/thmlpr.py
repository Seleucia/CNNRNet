from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Merge
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras import regularizers

def build_model(params):
   l2=regularizers.l2(0.001)
   l2_out=regularizers.l2(0.00001)
   dims=4096

   #########Left Stream######################
   lmodel = Sequential()
   lmodel.add(Dense(256, input_shape=(dims,),init='he_normal', W_regularizer=l2,activation='tanh'))
   lmodel.add(Dense(256,init='he_normal', W_regularizer=l2,activation='tanh'))
   lmodel.add(Dense(256,init='he_normal', W_regularizer=l2,activation='tanh'))

   #########Right Stream######################
   rmodel = Sequential()
   rmodel.add(Dense(256, input_shape=(dims,),init='he_normal', W_regularizer=l2,activation='tanh'))
   rmodel.add(Dense(256,init='he_normal', W_regularizer=l2,activation='tanh'))
   rmodel.add(Dense(256,init='he_normal', W_regularizer=l2,activation='tanh'))

   #########Merge Stream######################
   model = Sequential()
   model.add(Merge([lmodel, rmodel], mode='mul'))
   model.add(Dense(256,init='he_normal', W_regularizer=l2_out,activation='tanh'))

   model.add(Dense(256,init='he_normal'))
   model.add(Activation('linear'))

   model.add(Dense(params['n_output'],init='he_normal'))

   adagrad=Adagrad(lr=params['initial_learning_rate'], epsilon=1e-6)
   model.compile(loss='mean_squared_error', optimizer=adagrad)
   return model