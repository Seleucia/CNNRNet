import kcnnr
import dataset_loader
import numpy as np

def predict(test_set_x,params):
    model=kcnnr.build_model(params)
    model.load_weights(params['model_name'])
    #dataset parameters
    im_type=params["im_type"]
    nc =params["nc"]  # number of channcels
    size =params["size"]  # size = [480,640] orijinal size,[height,width]

    # learning parameters
    batch_size =params["batch_size"]
    n_test_batches = len(test_set_x)
    n_test_batches /= batch_size
    y_pred=[]
    print "Prediction on test images"
    for i in xrange(n_test_batches):
        Fx = test_set_x[i * batch_size: (i + 1) * batch_size]
        data_Fx = dataset_loader.load_batch_images(size, nc, "F", Fx,im_type)
        data_Sx = dataset_loader.load_batch_images(size, nc, "S", Fx,im_type)
        if(len(y_pred)==0):
            y_pred= model.predict([data_Fx, data_Sx])
        else:
            y_pred=np.concatenate((y_pred,model.predict([data_Fx, data_Sx])))
    return y_pred


