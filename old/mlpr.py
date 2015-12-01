import numpy
import theano
import theano.tensor as T
from theano import shared

from layers.HiddenLayer import HiddenLayer
from layers.OutputLayer import OutputLayer

# start-snippet-1

theano.config.exception_verbosity = 'high'

class MLPRNet(object):
    def __init__(self,
                 rng,
                 input,
                 batch_size,
                 nc,
                 Fx,
                 Sx):

        print("Done")

        self.layers = []
        self.params = []

        Flayer1 = HiddenLayer(
            rng,
            input=Fx,
            n_in=numpy.shape(Fx)[0]*numpy.shape(Fx)[0],
            n_out=200
        )

        Slayer1 = HiddenLayer(
            rng,
            input=Sx,
            n_in=numpy.shape(Sx)[0]*numpy.shape(Sx)[0],
            n_out=200
        )

        self.layers.append(Flayer1)
        self.layers.append(Slayer1)

        Flayer2 = HiddenLayer(
            rng,
            input=Flayer1.output,
            n_in=200,
            n_out=200
        )

        Slayer2 = HiddenLayer(
            rng,
            input=Slayer1.output,
            n_in=200,
            n_out=200
        )

        self.layers.append(Flayer2)
        self.layers.append(Slayer2)

        Clayer1 = HiddenLayer(
            rng,
            input=T.concatenate([Flayer2.output, Slayer2.output], axis=1),
            n_in=400,
            n_out=200
        )
        self.layers.append(Clayer1)

        Clayer2 = HiddenLayer(
            rng,
            input=Clayer1.output,
            n_in=200,
            n_out=200
        )
        self.layers.append(Clayer2)

        Clayer3 = HiddenLayer(
            rng,
            input=Clayer2.output,
            n_in=200,
            n_out=200
        )
        self.layers.append(Clayer3)

        layerOutput = OutputLayer(input=Clayer3.output, n_in=200, n_out=3)
        self.layers.append(layerOutput)

        for layer in zip(reversed(self.layers)):
            self.params.append(layer[0].W)
            self.params.append(layer[0].b)

        L1 = shared(0.)
        for param in self.params:
            L1 += (T.sum(abs(param[0]))+T.sum(abs(param[1])))
        self.L1=L1

        L2_sqr = shared(0.)
        for param in self.params:
            L2_sqr += (T.sum(param[0] ** 2)+T.sum(param[1] ** 2))
        self.L2_sqr=L2_sqr



        self.y_pred = self.layers[-1].y_pred
        self.mse = self.layers[-1].mse
        self.errors = self.layers[-1].errors


    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum

    def get_params(self):
        return numpy.concatenate([p.get_value().flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))

def train_model(params):
    rn_id=params["rn_id"]
    im_type=params["im_type"]
    nc =params["nc"]  # number of channcels
    size =params["size"]  # size = [480,640] orijinal size,[height,width]

    # Conv an Pooling parameters
    nkerns =params["nkerns"]
    kern_mat =params["kern_mat"]
    pool_mat =params["pool_mat"]

    # learning parameters
    batch_size =params["batch_size"]
    n_epochs =params["n_epochs"]
    initial_learning_rate =params["initial_learning_rate"]
    learning_rate_decay =params["learning_rate_decay"]
    squared_filter_length_limit =params["squared_filter_length_limit"]
    learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))
    lambda_1 = params["lambda_1"]  # regulizer param
    lambda_2 = params["lambda_2"]

    #### the params for momentum
    mom_start =params["mom_start"]
    mom_end = params["mom_end"]
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval =params["mom_epoch_interval"]

    # early-stopping parameters
    patience = params["patience"]  # look as this many examples regardless
    patience_increase = params["patience_increase"]  # wait this much longer when a new best is
    # found
    improvement_threshold = params["improvement_threshold"]  # a relative improvement of this much is



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
    params['mom_start']=0.0005  # the params for momentum
    params['mom_end']=0.0005    # the params for momentum
    params['mom_epoch_interval']=500 #for epoch in [0, mom_epoch_interval], the momentum increases linearly from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    params['initial_learning_rate']=0.0005
    params['learning_rate_decay']= 0.998
    params['squared_filter_length_limit']=15.0
    params['batch_size']=120
    params['n_epochs']=3000

    # dataset parameters
    params['dataset']="/home/coskun/PycharmProjects/data/rgbd_dataset_freiburg3_large_cabinet/"
    params['im_type']="gray"
    params['step_size']=[1,2,5,7,10,12,13,15,16,18,20,21,23,24,25]
    params['size']=[160, 120] #[width,height]
    params['nc']=1 #number of dimensions
    params['multi']=10 #ground truth location differences will be multiplied with this number
    params['test_size']=0.20 #Test size
    params['val_size']=0.20 #Test size

    train_model(params)
