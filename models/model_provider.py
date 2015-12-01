import dcnnr
import kcnnr


def get_model(params):
    if(params["model"]=="kcnnr"):
        model= kcnnr.build_model(params)
    if(params["model"]=="dcnnr"):
        model= dcnnr.build_model(params)
    return model

def get_model_pretrained(params):
    model=get_model(params)
    model.build()
    wd=params["wd"]
    model_name=wd+"/"+"models"+"/"+params['model_name']
    model.load_weights(model_name)
