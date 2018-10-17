'''
train a simple CNN-Capsule Network on the CIFAR10 small images dataset
'''

from __future__ import print_function
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.datasets import cifar10
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

