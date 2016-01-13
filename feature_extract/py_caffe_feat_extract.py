'''
author: ahmed osman
email : ahmed.osman99 AT GMAIL
'''

import sys
import helper.config as config
params=config.get_params()
sys.path.append(params["caffe"])
import caffe
import numpy as np
import os
import time
import show_images
import glob
import shutil
import helper.dt_utils as dt_utils


def reduce_along_dim(img , dim , weights , indicies):
   '''
   Perform bilinear interpolation given along the image dimension dim
   -weights are the kernel weights
   -indicies are the crossponding indicies location
   return img resize along dimension dim
   '''
   other_dim = abs(dim-1)
   if other_dim == 0:  #resizing image width
       weights  = np.tile(weights[np.newaxis,:,:,np.newaxis],(img.shape[other_dim],1,1,3))
       out_img = img[:,indicies,:]*weights
       out_img = np.sum(out_img,axis=2)
   else:   # resize image height
       weights  = np.tile(weights[:,:,np.newaxis,np.newaxis],(1,1,img.shape[other_dim],3))
       out_img = img[indicies,:,:]*weights
       out_img = np.sum(out_img,axis=1)

   return out_img


def cubic_spline(x):
   '''
   Compute the kernel weights
   See Keys, "Cubic Convolution Interpolation for Digital Image
   Processing," IEEE Transactions on Acoustics, Speech, and Signal
   Processing, Vol. ASSP-29, No. 6, December 1981, p. 1155.
   '''
   absx   = np.abs(x)
   absx2  = absx**2
   absx3  = absx**3
   kernel_weight = (1.5*absx3 - 2.5*absx2 + 1) * (absx<=1) + (-0.5*absx3 + 2.5* absx2 - 4*absx + 2) * ((1<absx) & (absx<=2))
   return kernel_weight

def contribution(in_dim_len , out_dim_len , scale ):
   '''
   Compute the weights and indicies of the pixels involved in the cubic interpolation along each dimension.

   output:
   weights a list of size 2 (one set of weights for each dimension). Each item is of size OUT_DIM_LEN*Kernel_Width
   indicies a list of size 2(one set of pixel indicies for each dimension) Each item is of size OUT_DIM_LEN*kernel_width

   note that if the entire column weights is zero, it gets deleted since those pixels don't contribute to anything
   '''
   kernel_width = 4
   if scale < 1:
       kernel_width =  4 / scale

   x_out = np.array(range(1,out_dim_len+1))
   #project to the input space dimension
   u = x_out/scale + 0.5*(1-1/scale)

   #position of the left most pixel in each calculation
   l = np.floor( u - kernel_width/2)

   #maxium number of pixels in each computation
   p = int(np.ceil(kernel_width) + 2)

   indicies = np.zeros((l.shape[0],p) , dtype = int)
   indicies[:,0] = l

   for i in range(1,p):
       indicies[:,i] = indicies[:,i-1]+1

   #compute the weights of the vectors
   u = u.reshape((u.shape[0],1))
   u = np.repeat(u,p,axis=1)

   if scale < 1:
       weights = scale*cubic_spline(scale*(indicies-u ))
   else:
       weights = cubic_spline((indicies-u))

   weights_sums = np.sum(weights,1)
   weights = weights/ weights_sums[:, np.newaxis]

   indicies = indicies - 1
   indicies[indicies<0] = 0
   indicies[indicies>in_dim_len-1] = in_dim_len-1 #clamping the indicies at the ends

   valid_cols = np.all( weights==0 , axis = 0 ) == False #find columns that are not all zeros

   indicies  = indicies[:,valid_cols]
   weights    = weights[:,valid_cols]

   return weights , indicies

def imresize(img , cropped_width , cropped_height):
   '''
   Function implementing matlab's imresize functionality default behaviour
   Cubic spline interpolation with antialiasing correction when scaling down the image.

   '''


   width_scale  = float(cropped_width)  / img.shape[1]
   height_scale = float(cropped_height) / img.shape[0]

   if len(img.shape) == 2: #Gray Scale Case
       img = np.tile(img[:,:,np.newaxis] , (1,1,3)) #Broadcast

   order   = np.argsort([height_scale , width_scale])
   scale   = [height_scale , width_scale]
   out_dim = [cropped_height , cropped_width]


   weights  = [0,0]
   indicies = [0,0]

   for i in range(0 , 2):
       weights[i] , indicies[i] = contribution(img.shape[ i ],out_dim[i], scale[i])

   for i in range(0 , len(order)):
       img = reduce_along_dim(img , order[i] , weights[order[i]] , indicies[order[i]])

   return img


def preprocess_image(img):
   '''
   Preprocess an input image before processing by the caffe module.


   Preprocessing include:
   -----------------------
   1- Converting image to single precision data type
   2- Resizing the input image to cropped_dimensions used in extract_features() matlab script
   3- Reorder color Channel, RGB->BGR
   4- Convert color scale from 0-1 to 0-255 range (actually because image type is a float the
       actual range could be negative or >255 during the cubic spline interpolation for image resize.
   5- Subtract the VGG dataset mean.
   6- Reorder the image to standard caffe input dimension order ( 3xHxW)
   '''
   img      = img.astype(np.float32)
   img      = imresize(img,224,224) #cropping the image
   img      = img[:,:,[2,1,0]] #RGB-BGR
   img      = img*255

   mean = np.array([103.939, 116.779, 123.68]) #mean of the vgg

   for i in range(0,3):
       img[:,:,i] = img[:,:,i] - mean[i] #subtracting the mean
   img = np.transpose(img, [2,0,1])
   return img #HxWx3

def caffe_extract_feats(parent_path,orijinal_img,conv_list,path_imgs , path_model_def , path_model ,ls_mx,batch_size):
   '''
   Function using the caffe python wrapper to extract 4096 from VGG_ILSVRC_16_layers.caffemodel model

   Inputs:
   ------
   path_imgs      : list of the full path of images to be processed
   path_model_def : path to the model definition file
   path_model     : path to the pretrained model weight
   WItH_GPU       : Use a GPU

   Output:
   -------
   features           : return the features extracted
   '''

   if params["WITH_GPU"]:
       caffe.set_mode_gpu()
   else:
       caffe.set_mode_cpu()
   print "loading model:",path_model
   caffe_net = caffe.Classifier(path_model_def , path_model , image_dims = (224,224) , raw_scale = None, channel_swap=(2,1,0),
                           mean = np.array([103.939, 116.779, 123.68]) )

   feats_fc7 = np.zeros((4096 , len(path_imgs)))
   feats_fc6 = np.zeros((4096 , len(path_imgs)))
   ls_mn={}
   for i in conv_list:
      ls_mn[i]=0

   for b in range(0 , len(path_imgs) , batch_size):
       list_imgs = []
       for i in range(b , b + batch_size ):
           if i < len(path_imgs):
               print path_imgs[i]
               img=caffe.io.load_image(path_imgs[i])

               if(params['orijinal_img']=="depth"):
                   img= np.dstack([img]*3)

               list_imgs.append( np.array( img ) ) #loading images HxWx3 (RGB)
           else:
               list_imgs.append(list_imgs[-1]) #Appending the last image in order to have a batch of size 10. The extra predictions are removed later..

       caffe_input = np.asarray([preprocess_image(in_) for in_ in list_imgs]) #preprocess the images

       caffe_net.forward(data = caffe_input)

       predictions_fc7 =caffe_net.blobs["fc7"].data.transpose()
       predictions_fc6 =caffe_net.blobs["fc6"].data.transpose()
       if i < len(path_imgs):
           for lm in conv_list:
              data=caffe_net.blobs[lm].data[:]
              ls_mx[lm]=np.maximum(np.max(data),ls_mx[lm])
              ls_mn[lm]+=np.mean(data)
              dt_utils.write_mid_features(data,parent_path+orijinal_img+"_"+lm+"/",path_imgs[b : b + batch_size])

           feats_fc7[:,b:i+1] = predictions_fc7
           feats_fc6[:,b:i+1] = predictions_fc6
           n = i+1
       else:
           n = min(batch_size , len(path_imgs) - b)
           feats_fc7[:,b:b+n] = predictions_fc7[:,0:n] #Removing extra predictions, due to the extra last image appending.
           feats_fc6[:,b:b+n] = predictions_fc6[:,0:n] #Removing extra predictions, due to the extra last image appending.
           for lm in conv_list:
              data=caffe_net.blobs[lm].data[0:n]
              ls_mx[lm]=np.maximum(np.max(data),ls_mx[lm])
              ls_mn[lm]+=np.mean(data)
              dt_utils.write_mid_features(data,parent_path+orijinal_img+"_"+lm+"/",path_imgs[b : b + batch_size])
           n += b
       print "%d out of %d done....."%(n ,len(path_imgs))
       if(params["run_mode"]==1):
           ls_mn=dict((key, value/batch_size) for (key, value) in ls_mn.iteritems())
           return (feats_fc6,feats_fc7,ls_mx,ls_mn)

   ls_mn=dict((key, value/feats.shape[1]) for (key, value) in ls_mn.iteritems())
   return (feats_fc6,feats_fc7,ls_mx,ls_mn)

if __name__ == '__main__':
   print "Main started"
   # parser = argparse.ArgumentParser()
   # parser.add_argument('--model_def_path',dest='model_def_path', type=str , help='Path to the VGG_ILSVRC_16_layers model definition file.')
   # parser.add_argument('--model_path', dest='model_path',type=str,  help='Path to VGG_ILSVRC_16_layers pretrained model weight file i.e VGG_ILSVRC_16_layers.caffemodel')
   # parser.add_argument('-i',dest='input_directory',help='Path to Directory containing images to be processed.')
   # parser.add_argument('--filter',default = None ,dest='filter', help='Text file containing images names in the input directory to be processed. If no argument provided all images are processed.')
   # parser.add_argument('--WITH_GPU', action='store_true', dest='WITH_GPU', help = 'Caffe uses GPU for feature extraction')
   # parser.add_argument('-o',dest='out_directory',help='Output directory to store the generated features')
   #
   # args = parser.parse_args()
   params['orijinal_img']="gray"
   path_model_def_file = "model/VGG_ILSVRC_16_layers_deploy.prototxt"
   path_model  = "model/VGG_ILSVRC_16_layers.caffemodel"
   filter_path = None
   WITH_GPU    = params["WITH_GPU"]
   out_directory = "data"
   print "arguments parsed"

   if not os.path.exists(path_model_def_file):
       raise RuntimeError("%s , Model definition file does not exist"%(path_model_def_file))

   if not os.path.exists(path_model):
       raise RuntimeError("%s , Path to pretrained model file does not exist"%(path_model))
   batch_size = 10
   mn6=0;
   mn7=0;
   mx6=-np.inf;
   mx7=-np.inf;
   counter=0
   img_counter=0
   fc_list=[]
   fc_list.append("fc6")
   fc_list.append("fc7")
   conv_list=[]
   conv_list.append("conv4_2")
   conv_list.append("conv5_2")
   ls_mx={}
   ls_mn={}
   for i in conv_list:
      ls_mx[i]=0
      ls_mn[i]=0
   params['run_mode']=1 #process checkY_testing
   orijinal_img=params['orijinal_img']
   for dir in params["dataset"]:
       if (dir == -1):
           continue
       for i in fc_list:
           break;
           new_dir= dir[0] + orijinal_img+"_"+i + "/"
           if(os.path.exists(new_dir)):
              shutil.rmtree(new_dir)
           os.makedirs(new_dir)
       for i in conv_list:
           break;
           new_dir= dir[0] + orijinal_img+"_"+ i + "/"
           if(os.path.exists(new_dir)):
              shutil.rmtree(new_dir)
           os.makedirs(new_dir)
       rgb_dir= dir[0] + orijinal_img + '/*.png'
       path_imgs =lst=glob.glob(rgb_dir)
       path_imgs=sorted(path_imgs)

       feats = caffe_extract_feats(dir[0],orijinal_img,conv_list,path_imgs, path_model_def_file, path_model,ls_mx,batch_size)
       img_counter+=len(path_imgs)
       fc6=feats[0]
       fc7=feats[1]
       ls_mx=feats[2]
       tmp_ls_mn=feats[3]
       ls_mn=dict((key, value+ls_mn[key]) for (key, value) in tmp_ls_mn.iteritems())

       mn6+=np.mean(fc6)
       mn7+=np.mean(fc7)
       mx6=np.maximum(mx6,np.max(fc6))
       mx7=np.maximum(mx7,np.max(fc7))

       dt_utils.write_features(fc6, path_imgs,dir[0] +orijinal_img+"_"+ fc_list[0] + "/")
       dt_utils.write_features(fc7, path_imgs,dir[0] +orijinal_img+"_"+ fc_list[1] + "/")
       print("data set converted %s"%(dir[0]))
       counter+=1

   print("#datasets: %s, #images: %s"%(counter,img_counter))
   print("fc6/fc7 mean: %s / %s, max: %s / %s"%(mn6/counter,mn7/counter,mx6,mx7))

   ls_mn=dict((key, value/counter) for (key, value) in ls_mn.iteritems())
   for i in conv_list:
      print("%s mean: %s, max: %s"%(i,ls_mn[i],ls_mx[i]))