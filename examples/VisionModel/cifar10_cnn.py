'''
train a simple deep CNN on the CIFAR10 small images dataset 

'''
import keras 
import ipdb 
from keras.datasets import cifar10 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
import os 

batch_size = 32 
num_classes = 10 
epoches = 100 
data_augmentation = True 
num_predictions = 20 
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("x_train/y_train shape is {}/{} and its sample num is {}/{}".format(x_train.shape, y_train.shape, x_train.shape[0], y_train.shape[0]))

ipdb.set_trace() 
# Convert class vevtors to binary class matrics 
y_train = keras.utils.to_categorical(y_train, num_classes) 
y_test = keras.utils.to_categorical(y_test, num_classes) 

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# intiate RMSprop optimizer 
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_train /= 255
y_train /= 255


