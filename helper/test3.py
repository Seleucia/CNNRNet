import utils
import config as config
import tum_dataset_loader as tdl
import utils
params=config.get_params()

print "Mean Compute Started...."
tdl.compute_mean(params)
print "Gray scale conver started...."
utils.convert_to_grayscale(params)