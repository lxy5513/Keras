'''
Train a resnet on the cifer10 dataset 

RetNet v1 
deep residual learning for image recognition 

resNet v2 
Identity Mappings in deep resdidual network 
'''
 
import keras  
from keras.layers import Dense, Conv2D, BatchNormalization, Activation 
from keras.layers import AveragePooling2D, Input, Flatten 
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import LearningRateScheduler 
from keras.preprocessing.image import ImageDataGenerator 
from keras.regularizers import l2 
from keras.models import Model 
from keras import backend as K 
from keras.datasets import cifar10 
import numpy as np 
import os 
import ipdb

#  training params 
epochs = 200
batch_size = 32 
data_augmentation = True 
num_classes = 10 

# subtracting pixel mean improves accuracy ------------???
subtract_pixel_mean = True 

n = 3 

# 1 --> resnet1  2 --> resnet2
version = 1
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2 

# model name (depth version)
model_type = 'ResNet{}v{}'.format(depth, version)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
ipdb.set_trace() 

#input image dimentions 
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean 
    x_test -= x_train_mean 

