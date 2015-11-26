import utils
import config as config
import plot_data
params=config.get_params()
params["log_file"]="log_12_47_03_872503.txt"
list_val=utils.log_read("VAL",params)
plot_val(list_val,params["log_file"].replace(".txt",".png"))
#utils.convert_to_grayscale(params)
print "Images are converted to gray scale"