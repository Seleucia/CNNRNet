import helper.utils as utils
import helper.config as config
import plot_data

params=config.get_params()
params["log_file"]="dcnnr_dr_0_20-24-01-476440.txt"

model="VAL"
list_val=utils.log_read(model,params)
plot_data.plot_val(list_val,params["wd"]+"/"+"logs/img/"+params["log_file"].replace(".txt",".png"))

list_val=utils.log_read_train(params)
plot_data.plot_val_train(list_val,params["wd"]+"/"+"logs/img/"+params["log_file"].replace(".txt",".png"),-1)
