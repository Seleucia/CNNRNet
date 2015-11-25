import PIL
import numpy as np
import pylab
import matplotlib.cm as cm
from matplotlib.mlab import PCA



def show_layer_activations(caffe_net,layer,idx):
    #caffe_net.blobs["conv2_1"].data[1][70]
    i=0
    f = pylab.figure()
    ax = pylab.gca()
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    grd=[4,4]
    counter=0
    top_n=grd[0]*grd[1]
    n_idx=np.argpartition(np.mean(np.mean(caffe_net.blobs[layer].data[0],axis=1),axis=1),-top_n)[-top_n:]
    results = PCA(np.reshape(caffe_net.blobs[layer].data[idx],(512,28*28)))
    for i in n_idx:
        img=caffe_net.blobs[layer].data[idx][i]
        f.add_subplot(grd[0], grd[1], counter)  # this line outputs images on top of each other
        # f.add_subplot(1, 2, n)  # this line outputs images side-by-side
        pylab.imshow(img,cmap=cm.Greys_r)
        counter+=1
        if(counter>=100):
            break
    pylab.show()
    print("ok")
