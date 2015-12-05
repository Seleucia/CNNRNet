import utils
import config as config
import tum_dataset_loader as tdl
import data_loader as dl
import dt_utils as dt
import utils
params=config.get_params()

#ds=dl.load_data_with_id(params,17)
print("okayuu")
print "Mean Compute Started...."
dt.check_missin_values(params)
#tdl.compute_mean(params)
#print "Gray scale conver started...."
#utils.convert_to_grayscale(params)