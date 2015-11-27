import utils
import config as config
import tum_dataset_loader as tdl
params=config.get_params()

tdl.compute_mean(params)