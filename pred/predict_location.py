import kcnnr
import dcnnr
import helper.dt_utils as du
import numpy as np

def predict(model_type,test_set_x,params):
    if(model_type=="kccnr"):
        model=kcnnr.build_model(params)

    if(model_type=="kccnr"):
        model=dcnnr.build_model(params)

    model.build()
    wd=params["wd"]
    model_name=wd+"/"+"models"+"/"+params['model_name']
    model.load_weights(model_name)

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


