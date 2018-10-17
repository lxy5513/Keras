'''
IMDB movies web 
'''

import numpy as np 
from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding 
from keras.layers import LSTM
from keras.layers import Bidirectional 
from keras.datasets import imdb 

# 选择频率最高的20000个词
max_feature = 20000 
# cut texts after this of words 
maxlen = 100 
batch_size = 32 

print('Loading data ... ')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature) 
print( 'The x_train/x_test sequences is {}/{}'.format(len(x_train), len(x_test)))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
y_train = sequence.pad_sequences(y_train, maxlen=maxlen)
print('The shape of x_train/x_test is {}/[]'.format(x_train.shape, y_train.shape))
y_train = np.array(y_train) 
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5)) 
model.add(Dense(1, activation='sigmoid')) 

model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) 

print('Training ...')
model.fit(x_train, y_train, 
        batch_size=batch_size,
        epochs=4,
        validation_data=[x_test, y_test]
        ) 

