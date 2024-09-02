
import numpy
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
import time

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Import the dataset
dataset = pd.read_csv("../Linear_Data.csv", header=None)
df = dataset.values
data = df[:,0:1]
label =  df[:,1]
sc=MinMaxScaler()
data = sc.fit_transform(data)
X_train, X_test, Y_train, Y_test = train_test_split(data, label, 
                                                    test_size=0.2)

# Now we build the model
neural_network = Sequential() # create model
neural_network.add(Dense(5, input_dim=1)) # hidden layer
neural_network.add(BatchNormalization())
neural_network.add(Activation('sigmoid'))
neural_network.add(Dropout(0.1))
neural_network.add(Dense(1)) # output layer

neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
neural_network_fitted = neural_network.fit(X_train, Y_train, epochs=200, verbose=1, 
                                           batch_size=32, initial_epoch=0,)
                                       
