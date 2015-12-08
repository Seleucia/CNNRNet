import dcnnr
import kcnnr
import pcnnr
import bncnnr
import mlpr
import bnmlpr
import dmlpr
import thmlpr


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
    return model

def get_model_pretrained(params):
    model=get_model(params)
    model.build()
    model_name=params['model_file']+params['model_name']
    model.load_weights(model_name)
    return model
