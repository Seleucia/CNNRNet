import numpy as np
import helper.utils as u
import helper.dt_utils as du
from models import model_provider


def predict(test_set_x,params):
    if(params['patch_use']== 1):
        y_pred=predict_on_patch(test_set_x,params)
    else:
        y_pred=predict_on_fullimage(test_set_x,params)
    return y_pred


def predict_on_fullimage(test_set_x,params):
    model= model_provider.get_model_pretrained(params)
    # learning parameters
    batch_size =params["batch_size"]
    n_test_batches = len(test_set_x)
    ash=n_test_batches%batch_size
    if(ash>0):
        test_set_x=np.vstack((test_set_x,np.tile(test_set_x[-1],(batch_size-ash,1))))
        n_test_batches = len(test_set_x)

    n_test_batches /= batch_size
    y_pred=[]
    print "Prediction on test images"
    for i in xrange(n_test_batches):
        Fx = test_set_x[i * batch_size: (i + 1) * batch_size]
        data_Fx = du.load_batch_images(params, "F", Fx)
        data_Sx = du.load_batch_images(params,"S", Fx)
        if(len(y_pred)==0):
            y_pred= model.predict([data_Fx, data_Sx])
        else:
            y_pred=np.concatenate((y_pred,model.predict([data_Fx, data_Sx])))
    if(ash>0):
        y_pred= y_pred[0:-(batch_size-ash)]
    return y_pred


def predict_on_patch(test_set_x,params):

    model= model_provider.get_model_pretrained(params)

    # learning parameters
    batch_size =params["batch_size"]
    n_test_batches = len(test_set_x)
    ash=n_test_batches%batch_size
    if(ash>0):
        test_set_x=np.vstack((test_set_x,np.tile(test_set_x[-1],(batch_size-ash,1))))
        n_test_batches = len(test_set_x)

    n_test_batches /= batch_size
    y_pred=[]
    print "Prediction on test images"
    for i in xrange(n_test_batches):
        Fx = test_set_x[i * batch_size: (i + 1) * batch_size]
        n_patch=params["n_patch"]
        pred=np.zeros(params['n_output'])
        for patch_index in xrange(n_patch):
            patch_loc=u.get_patch_loc(params)
            data_Fx = du.load_batch_images(params, "F", Fx,patch_loc)
            data_Sx = du.load_batch_images(params,"S", Fx,patch_loc)
            pred=np.add(pred, model.predict([data_Fx, data_Sx]))

        pred/=n_patch
        if(len(y_pred)==0):
            y_pred= pred
        else:
            y_pred=np.concatenate((y_pred,pred))

    if(ash>0):
        y_pred= y_pred[0:-(batch_size-ash)]
    return y_pred



