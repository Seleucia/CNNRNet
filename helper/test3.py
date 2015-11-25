import utils
import helper.config as config

params=config.get_params()
utils.convert_to_grayscale(params)
print "Images are converted to gray scale"