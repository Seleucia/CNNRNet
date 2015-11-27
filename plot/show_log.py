import helper.utils as utils
import helper.config as config
import plot_data

params=config.get_params()
params["log_file"]="log_12_47_03_872503.txt"

list_val=utils.log_read_train(params)
#plot_data.plot_val(list_val,params["wd"]+"/"+"predictions/"+params["log_file"].replace(".txt",".png"))

plot_data.plot_val_train(list_val,params["wd"]+"/"+"logs/img/"+params["log_file"].replace(".txt",".png"),-1)
