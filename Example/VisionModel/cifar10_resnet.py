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
from keras.callbacks import ReduceLROnPlateau # Reduce learning rate when a metric has stopped improving
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
from keras.datasets import cifar10
import numpy as np
import os
import ipdb
import logging
import sys
import logging.handlers
import datetime

# set log
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# set all.log
path_log = os.path.abspath(os.path.join(sys.path[0], '../log/cifar10_resnet'))
if not(os.path.exists(path_log)):
    os.makedirs(path_log)

all_log_path = path_log + '/all.log'
rf_handle = logging.handlers.TimedRotatingFileHandler(all_log_path, when='midnight', interval=1,
                                                   backupCount=7, atTime=datetime.time(0, 0, 0, 0 ))
rf_handle.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s")
)

# set error.log
error_log_path = path_log + '/error.log'
f_handler = logging.FileHandler(error_log_path)
f_handler.setLevel(logging.ERROR)
f_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s")
)
logger.addHandler(rf_handle)
logger.addHandler(f_handler)

#  training params
epochs = 100
batch_size = 32
data_augmentation = True
num_classes = 10

# subtracting pixel mean improves accuracy ------------???
subtract_pixel_mean = True
n = 6

# 1 --> resnet1  2 --> resnet2
version = 1
if version == 1:
    depth = n * 6 + 2
    model_base = 'resnet1'
elif version == 2:
    depth = n * 9 + 2
    model_base = 'resnet2'

logger.info("Model paras ----------> epochs: {} depth: {} version {}".format(epochs, depth, model_base))


# model name (depth version)
model_type = 'ResNet{}v{}'.format(depth, version)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#  ipdb.set_trace()

#input image dimentions
input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape: {}\t y_train shape: {}'.format(x_train.shape, y_train.shape))
#  logger.info('x_train shape: {}\t y_train shape: {}'.format(x_train.shape, y_train.shape))

# conert class  vectors to binary class metrices  将一维向量 转化成 十维向量（one-hot)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    '''
    Learning rate is scheduled to be reduced after 80 120 160 180 epoch
    '''
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch >120:
        lr *= 1e-2
    elif epoch > 100:
        lr *- 1e-1
    print('LearningRate is ', lr)
    return lr

def resnet_layer(inputs,
                num_filters = 16,
                kernel_size = 3,
                strides =  1,
                activation = 'relu',
                batch_normlization = True,
                conv_first = True):
    '''
    inputs (tensor) input tensor from input image or previous layer
    num_filters (int) Conv2D number of filters
    kernel_size (int) Conv2D square kelnel dimensions
    strides (int) Conv2D square stride dimension
    conv_first (bool) conv-bn-activation(True) bn-conv-activation(False)

    return
        x(tensor) tensor or input to next layer
    '''
    # 二维卷积层，即对图像的空域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。
    # 例如input_shape = (128,128,3)代表128*128的彩色RGB图像（data_format='channels_last'） 并不是处理二维图片。
    conv = Conv2D(num_filters,
                kernel_size=kernel_size,
                strides = strides,
                padding = 'same',
                kernel_initializer = 'he_normal',
                kernel_regularizer = l2(1e-4)
                )
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normlization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normlization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    '''
    stacks of 2 * (3*3) Conv2D-BN-RELU
    Last relu  is after the shortcut connection
    At the begining of each stage the feature map size id halved (downsample)
    by a convolutional layer with strides=2 while the number of filter is doubled
    with each stage, the layers have same number filers and the same number of num_filters
    Feature maps sizes:
    stage 0: 32*32 16
    stage 1: 16*16 32
    stage 2: 8*8 64
    the number of parameters is approx the same as table 6 of a
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    input_shape (tensor) shape of image tensor
    depth (int) number of core convoluntional layers
    num_classes (int) number of class

    # return
        model(Model) keras  model instance
    '''
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n + 2')

    # Start model definition
    num_filters = 16 # 表明生成的feature depth 是多少
    num_res_block = int((depth-2)/6) # block的数量 each resnet block have 6 layers

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_block):
            strides  = 1
            if stack > 0 and res_block == 0:
                strides = 2 #downsample 保证残差前后的维度一样
            y = resnet_layer(inputs=x,
                            num_filters=num_filters,
                            strides=strides,
                            )
            y = resnet_layer(inputs=y,
                            num_filters=num_filters,
                            activation=None,
                            )
            if stack > 0 and res_block == 0: # first layer but not first stack
                # linear projection residual shortcut connnection to match
                # change dims
                x = resnet_layer(inputs=x,
                                num_filters=num_filters,
                                kernal_size=1,
                                strides=strides,
                                activation=None,
                                batch_normlization=False)
            x = keras.layers.add([x,y])
            x = Activation('relu')(x)
        num_filters *= 2

        # add classifier on top
        # v1 does not use BN after last shortcut connection on RELU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

def resnet_v2(input_shape, depth, num_classes=10):
    '''
    ResNet version 2 model builder
    stacks of (1*1)-(3*3)-(1*1) BN-RELU-Conv2D or also known bottleneck layer
    First shortcut connection per layer is 1*1 Conv2D
    second and onwards shortcut connection is identity
    At the beginning of each stage, the feature map size is halved(dawnsample)
    by a conv layer with strides=2, the layers have same number filters and same filter map size

    Feature map size:
        conv1 : 32*32 16
        stage0 32*32 64
        stage1 32*32 128
        stage2 32*32 256

    '''
    if (depth-2) % 9 != 0:
        raise ValueError('depth should be 9n+2')
    num_filters_in = 16
    num_res_blocks = int((depth-2) / 9)
    inputs = Input(shape=input_shape)

    # v2 performs Conv2D with BN-RELU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Instantiate the stack of units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normlization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normlization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2 # downsample

            # bottleneck residual unit
            y = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_in,
                    kernel_size=1,
                    strides=strides,
                    activation=activation,
                    batch_normlization=batch_normlization,
                    conv_first=False
                    )
            y = resnet_layer(
                    inputs=y,
                    num_filters=num_filters_in,
                    conv_first=False
                    )
            y = resnet_layer(
                    inputs=y,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    conv_first=False
                    )
            if res_block == 0:
                # linear projection residual shortcut connection to match ---change dims by num_filters_out
                x = resnet_layer(
                        inputs=x,
                        num_filters=num_filters_out,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normlization=False
                        )
            x = keras.layers.add([x, y])
        num_filters_in = num_filters_out
    # Add classfier on top
    # v2 has BN-RELU before Pooling

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=lr_schedule(0)),
        metrics=['accuracy']
        )

model.summary()


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print('save model location is: ', filepath)

# Prepare callbacks for model saving and for learning rate adjustment.  period = 10 每隔十次做一次判断
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             period=10)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        # validation_split=0.0
        )

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

    # 打印history
    print('history ----------------------------> ', history.history['val_acc'])
    logger.info('validation accacury: ')
    logger.info(history.history['val_acc'])


# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
logger.info('ASSET: test loss is {} test accuracy is {}'.format(scores[0], scores[1]))











'''
    ResNet结构
它使用了一种连接方式叫做“shortcut connection”，
    残差指的是什么？
其中ResNet提出了两种mapping：一种是identity mapping，指的就是图1中”弯弯的曲线”，另一种residual mapping，指的就是除了”弯弯的曲线“那部分，所以最后的输出是 y=F(x)+x
identity mapping顾名思义，就是指本身，也就是公式中的xx，而residual mapping指的是“差”，也就是y−xy−x，所以残差指的就是F(x)F(x)部分。
对于“随着网络加深，准确率下降”的问题，Resnet提供了两种选择方式，也就是identity mapping和residual mapping，如果网络已经到达最优，继续加深网络，residual mapping将被push为0，只剩下identity mapping，这样理论上网络一直处于最优状态了，网络的性能也就不会随着深度增加而降低了。
'''

