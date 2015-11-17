import os
import sys
import timeit
import cPickle
import numpy

import theano
import theano.tensor as T
import  model_saver
import convTest

from logistic_depth_sgd import LogisticRegression,dataset_loader

theano.config.exception_verbosity='high'



def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model


    #This must be loaded also
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # labels, presented as 1D vector of [int] labels
                        # [int] labels
    n_hidden=500
    rng = numpy.random.RandomState(1234)
    size = 32, 24
    classifier = convTest.MLP(
        rng=rng,
        input=x,
        n_in=size[0]*size[1],
        n_hidden=n_hidden,
        n_out=3
    )
    classifier.set_model_params(model_saver.load_model("3"))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred)

    datasets = dataset_loader.load_tum_data()
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y=test_set_y.get_value()

    predicted_values = predict_model(test_set_x[:10])
    actual_values = test_set_y[:10]
    res=T.dot(test_set_x[:1], classifier.params[0]) + classifier.params[1]
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values
    print actual_values


predict()