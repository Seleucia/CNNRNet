import os
import sys
import timeit
import cPickle
import numpy
import datetime

import theano
import theano.tensor as T


def save_model(ext,params):
    i=0
    with file('models/'+str(ext)+'_model.pkl', 'wb') as f:
        p = cPickle.Pickler(f,protocol=cPickle.HIGHEST_PROTOCOL)
        p.fast = True
        p.dump(params)


def save_pred(ext,pred):
    i=0
    with file('predictions/'+str(ext), 'wb') as f:
        p = cPickle.Pickler(f,protocol=cPickle.HIGHEST_PROTOCOL)
        p.fast = True
        p.dump(pred)

def load_pred(pred_name):
    i=0
    with file('predictions/'+pred_name, 'rb') as f:
        return cPickle.load(f)




def save_garb(obj):
    tm= str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)+"-"+str(datetime.datetime.now().second)
    with file('garb/'+tm+'_obj.pkl', 'wb') as f:
        p = cPickle.Pickler(f,protocol=cPickle.HIGHEST_PROTOCOL)
        p.fast = True
        p.dump(obj)


def save_partitions(ext,partitions):
    tm= str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)+"-"+str(datetime.datetime.now().second)
    with file('data/'+str(ext)+"_"+tm+'_sets.pkl', 'wb') as f:
        p = cPickle.Pickler(f)
        p.fast = True
        p.dump(partitions)

def load_partitions(name):
    with file('data/'+name, 'rb') as f:
        return cPickle.load(f)

def load_model(name):
    with file('models/'+name, 'rb') as f:
        return cPickle.load(f)

