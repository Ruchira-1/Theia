
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import time 
import sys
import keras

iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.5, random_state=2)

n_features = X.shape[1]
n_classes = Y.shape[1]

# Split the data set into training and testing
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=2)

# n_features = X.shape[1]
# n_classes = Y.shape[1]

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout

start_time = time.clock()
model = Sequential()
model.add(Dense(4, input_dim=4, init='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3, init='normal'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train,batch_size=32,
                                 epochs=50,
                                 verbose=1,
                                 validation_data=(X_test, Y_test)
