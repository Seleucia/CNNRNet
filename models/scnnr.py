from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Merge,Siamese,Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.optimizers import SGD
from keras import regularizers
import sys

sys.setrecursionlimit(50000)

#########Model without droupout######################
def build_model():
   input_model_1 = Sequential()
   input_model_1.add(Dense(input_dim=10, output_dim=10))
   input_model_1.add(Dense(input_dim=10, output_dim=10))
   input_model_1.add(Dropout(0.5))

   input_model_2 = Sequential()
   input_model_2.add(Dense(input_dim=10, output_dim=10))
   input_model_2.add(Dense(input_dim=10, output_dim=10))
   input_model_2.add(Dropout(0.5))

   inputs = [input_model_1, input_model_2]

   layer = Dense(input_dim=10, output_dim=5)

   model = Sequential()
   model.add(Siamese(layer ,inputs, 'sum'))
   model.add(Dense(input_dim=10, output_dim=10))
   model.add(Dense(input_dim=10, output_dim=10))
   model.compile(loss='mse', optimizer='sgd')

   l2=regularizers.l2(0.01)
   l2_out=regularizers.l2(0.001)

build_model()