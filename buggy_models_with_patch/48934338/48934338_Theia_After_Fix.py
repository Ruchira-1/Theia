
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
import numpy as np
import keras
from keras import optimizers
import time 
import sys
from sklearn.preprocessing import  MinMaxScaler


# fix random seed for reproducibility
seed = 7
#datapoints
X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)
sc = MinMaxScaler()
X = sc.fit_transform(X)

#model
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(30, init='uniform',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))

#training
sgd = optimizers.SGD(lr=0.001)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
print(X[0])
model.fit(X, y, epochs=1000)

