import dataset_loader
import numpy
theano.config.exception_verbosity = 'high'

size=[160,120]
n_visible=size[0] * size[1]
n_hidden=2000
batch_size=20
dataset="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
X_Pairs=dataset_loader.load_pairs(dataset,step_size=[])


initial_W = numpy.asarray(
               numpy.random.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                )
            )

W_prime=numpy.transpose(initial_W)

bvis = numpy.asarray(numpy.zeros(
                    n_visible
                ))


b_prime = bvis

b = numpy.zeros(n_hidden)

def sigmoid(x):
  return 1 / (1 + numpy.exp(-x))

def get_hidden_values(input,W,b):
        """ Computes the values of the hidden layer """
        return sigmoid(numpy.dot(input, W) + b)

def get_reconstructed_input(hidden,W_prime,b_prime):
        return sigmoid(numpy.dot(hidden, W_prime) + b_prime)

i=0
Fx = X_Pairs[i * batch_size: (i + 1) * batch_size]
Sx = dataset_loader.load_batch_images(size,1, "F", Fx)
Fx = dataset_loader.load_batch_images(size,1, "S", Fx)

tilde_x =Sx
y = get_hidden_values(tilde_x,initial_W,b)
z = get_reconstructed_input(y,W_prime,b_prime)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
L = - numpy.sum(Fx * numpy.log(z) + (1 - Fx) * numpy.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
cost = numpy.mean(L)



print "ok"