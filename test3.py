import os
import sys
import timeit
import numpy
import utils
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import dataset_loader
import model_saver
from CNN_RegressionV3 import CNNRNet

dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
rng = numpy.random.RandomState(23455)
# size = [480,640] orijinal size
rn_id=1
size = [120, 160]
nc = 1  # number of channcels
nkerns = [20, 50]
nkern1_size = [5, 5]
nkern2_size = [5, 5]

npool1_size = [2, 2]
npool2_size = [2, 2]

batch_size = 30

multi = 10

learning_rate = 0.0005
n_epochs = 400

initial_learning_rate = 0.0005
learning_rate_decay = 0.998
squared_filter_length_limit = 15.0
n_epochs = 3000
learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))

#### the params for momentum
mom_start = 0.5
mom_end = 0.99
# for epoch in [0, mom_epoch_interval], the momentum increases linearly
# from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
mom_epoch_interval = 500
mom_params = {"start": mom_start,
              "end": mom_end,
              "interval": mom_epoch_interval}

lambda_1 = 0.01  # regulizer param
lambda_2 = 0.01

datasets = dataset_loader.load_tum_dataV2(dataset,rn_id,multi)

X_train, y_train = datasets[0]
X_val, y_val = datasets[1]
X_test, y_test = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = len(X_train)
n_valid_batches = len(X_val)
n_test_batches = len(X_test)

n_train_batches /= batch_size
n_valid_batches /= batch_size
n_test_batches /= batch_size

epoch = T.scalar()
Fx = T.matrix(name='Fx_input')  # the data is presented as rasterized images
Sx = T.matrix(name='Sx_input')  # the data is presented as rasterized images
y = T.matrix('y')  # the output are presented as matrix 1*3.

Fx_inp = T.matrix(name='Fx_inp')  # the data is presented as rasterized images
Sx_inp = T.matrix(name='Sx_inp')  # the data is presented as rasterized images
y_inp = T.matrix('y_inp')

print '... building the model'

cnnr = CNNRNet(rng, input, batch_size, nc, size, nkerns,
               nkern1_size, nkern2_size,
               npool1_size, npool2_size,
               Fx, Sx)

