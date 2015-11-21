import numpy
import theano
import theano.tensor as T
import  model_saver
import dataset_loader
from CNN_RegressionV3 import CNNRNet

theano.config.exception_verbosity='high'

def predict(test_set_x,model_name,batch_size,size):
    Fx = T.matrix(name='Fx_input')  # the data is presented as rasterized images
    Sx = T.matrix(name='Sx_input')  # the data is presented as rasterized images
    Fx_inp = T.matrix(name='Fx_inp')  # the data is presented as rasterized images
    Sx_inp = T.matrix(name='Sx_inp')  # the data is presented as rasterized images
    rng = numpy.random.RandomState(23455)
    # size = [480,640] orijinal size,[height,width]
    nc = 1  # number of channcels
    nkerns = [30, 40]
    nkern1_size = [5, 5]
    nkern2_size = [5, 5]

    npool1_size = [2, 2]
    npool2_size = [2, 2]

    print '... building the model'

    cnnr = CNNRNet(rng, input, batch_size, nc, size, nkerns,
                   nkern1_size, nkern2_size,
                   npool1_size, npool2_size,
                   Fx, Sx)

    cnnr.load(model_name)
    #cnnr.set_params(model_saver.load_model(model_name))
    print "Model parameters loaded"

     # create a function to compute the mistakes that are made by the model
    predict_model = theano.function(
        [Fx_inp, Sx_inp],
        cnnr.y_pred,
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp,
        }
    )

    n_test_batches = len(test_set_x)
    n_test_batches /= batch_size
    y_pred=[]
    print "Prediction on test images"
    for i in xrange(n_test_batches):
        Fx = test_set_x[i * batch_size: (i + 1) * batch_size]
        data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx)
        data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx)
        if(len(y_pred)==0):
            y_pred= predict_model(data_Fx, data_Sx)
        else:
            y_pred=numpy.concatenate((y_pred,predict_model(data_Fx, data_Sx)))

    return y_pred
