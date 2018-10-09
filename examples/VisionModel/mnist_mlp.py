'''
Trains a simple deep NN on the MNIST dataset
'''

from __future__ import print_function

import ipdb
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#  ipdb.set_trace(context=6)

# 这是干什么？-----------------------------
x_train /= 255
x_test /= 255

# Convert calss vectors to binary class metrices-------------------?
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# 打印出模型概况，它实际调用的是keras.utils.print_summary
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 512)               401920
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
'''
model.summary()


# what meaning of matrics??  列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是metrics=['accuracy']
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss', score[0])
print('Test accuracy: ', score[1])

'''
By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.

verbose=0 will show you nothing (silent)

verbose=1 will show you an animated progress bar like this:

progres_bar <===================================>

verbose=2 will just mention the number of epoch like this:

1/4

感悟：
    1. metrics be used to evaluate the performance of train model 
    2. verbose be used to 'see' the training progress for each epoch
    3. MLP(Multilayer Perceptron) 这里使用的是sequential序列模型，有5个网络层，包括两个用于减小过拟合的dropout层
    
'''
