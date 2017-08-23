import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pylab import *

# specify path
caffe_path = '/media/ntfs-2/ZWJ/caffe-master/'
network_path = '/media/ntfs-2/ZWJ/caffe-master/examples/mnist/kaggle/nin_test.prototxt'
model_path = '/media/ntfs-2/ZWJ/caffe-master/examples/mnist/kaggle/_iter_24000.caffemodel'
dec_values_path = '/home/zhao/Desktop/kaggle/mnist/'
 
# enter caffe path, add pycaffe path, CSI_func path to environment 
os.chdir(caffe_path)
sys.path.insert(0, caffe_path + 'python')

import caffe

# load the true labels of testset, solver and pick a gpu
net = caffe.Net(network_path, model_path,caffe.TEST)

################################################################################################### 
# set the training parameters

caffe.set_device(0)  # we can use 0,2,3
caffe.set_mode_gpu()


# solver parameters
test_iterations = 280
testset_batchsize = 100


###################################################################################################

# predicting
values = []

for test_it in range(test_iterations):

	net.forward()
		
	dec_value_argmax = net.blobs['softmaxoutput'].data.argmax(1)

	dec_value_argmax=dec_value_argmax.tolist()
	values += dec_value_argmax

np.savetxt( dec_values_path + 'result.txt', values, fmt='%d' )
