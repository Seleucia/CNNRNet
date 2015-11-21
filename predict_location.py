import numpy
import theano
import theano.tensor as T
import  model_saver
import dataset_loader
from cnnr import CNNRNet

theano.config.exception_verbosity='high'

def predict(test_set_x,params):

    Fx = T.matrix(name='Fx_input')  # the data is presented as rasterized images
    Sx = T.matrix(name='Sx_input')  # the data is presented as rasterized images
    Fx_inp = T.matrix(name='Fx_inp')  # the data is presented as rasterized images
    Sx_inp = T.matrix(name='Sx_inp')  # the data is presented as rasterized images
    rng = numpy.random.RandomState(23455)

    #dataset parameters
    im_type=params["im_type"]
    nc =params["nc"]  # number of channcels
    size =params["size"]  # size = [480,640] orijinal size,[height,width]

    # Conv an Pooling parameters
    nkerns =params["nkerns"]
    kern_mat =params["kern_mat"]
    pool_mat =params["pool_mat"]

    # learning parameters
    batch_size =params["batch_size"]


    print '... building the model'

    cnnr = CNNRNet(rng, input, batch_size, nc, size, nkerns,
                   kern_mat[0], kern_mat[1],
                   pool_mat[0], pool_mat[1],
                   Fx, Sx)

    cnnr.load(params['model_name'])
    #cnnr.set_params(model_saver.load_model(model_name))
    print "Model parameters loaded"

     # create a function to compute the mistakes that are made by the model
    predict_model = theano.function(
        [Fx_inp, Sx_inp],
        cnnr.y_pred,
        givens={
            Fx: Fx_inp,
            Sx: Sx_inp,
        },allow_input_downcast=True
    )

    n_test_batches = len(test_set_x)
    n_test_batches /= batch_size
    y_pred=[]
    print "Prediction on test images"
    for i in xrange(n_test_batches):
        Fx = test_set_x[i * batch_size: (i + 1) * batch_size]
        data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx,im_type)
        data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx,im_type)
        if(len(y_pred)==0):
            y_pred= predict_model(data_Fx, data_Sx)
        else:
            y_pred=numpy.concatenate((y_pred,predict_model(data_Fx, data_Sx)))

    return y_pred
