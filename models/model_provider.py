import dcnnr
import kcnnr
import pcnnr
import bncnnr
import mlpr
import bnmlpr
import dmlpr
import thmlpr
import conv4mlpr
import schcnnr
import schcnnr_in
import schcnnr_agr

def get_model(params):
    if(params["model"]=="kcnnr"):
        model= kcnnr.build_model(params)
    if(params["model"]=="dcnnr"):
        model= dcnnr.build_model(params)
    if(params["model"]=="bncnnr"):
        model= bncnnr.build_model(params)
    if(params["model"]=="mlpr"):
        model= mlpr.build_model(params)
    if(params["model"]=="bnmlpr"):
        model= bnmlpr.build_model(params)
    if(params["model"]=="dmlpr"):
        model= dmlpr.build_model(params)
    if(params["model"]=="thmlpr"):
        model= thmlpr.build_model(params)
    if(params["model"]=="pcnnr"):
        model= pcnnr.build_model(params)
    if(params["model"]=="conv4mlpr"):
        model= conv4mlpr.build_model(params)
    if(params["model"]=="schcnnr"):
        model= schcnnr.build_model(params)
    if(params["model"]=="schcnnr_in"):
        model= schcnnr_in.build_model(params)
    if(params["model"]=="schcnnr_agr"):
        model= schcnnr_agr.build_model(params)
    return model

def get_model_pretrained(params):
    model=get_model(params)
    model.build()
    model_name=params['model_file']+params['model_name']
    model.load_weights(model_name)
    return model
