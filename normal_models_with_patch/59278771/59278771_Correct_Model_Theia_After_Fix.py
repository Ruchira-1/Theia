import numpy  
import pandas 
from keras.models import Sequential 
from keras.layers import Dense, Activation, BatchNormalization,Dropout
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import Pipeline 
import time
from sklearn.preprocessing import MinMaxScaler
import keras

import sys
# fix random seed for reproducibility 
seed = 7 
numpy.random.seed(seed) 
# load dataset 
dataframe = pandas.read_csv("../iris.csv", header=None) 
dataset = dataframe.values 
X = dataset[:,0:4].astype(float) 
Y = dataset[:,4] 

# encode class values as integers 
encoder = LabelEncoder() 
encoder.fit(Y) 
encoded_Y = encoder.transform(Y) 
sc = MinMaxScaler()
X = sc.fit_transform(X)
# convert integers to dummy variables (i.e. one hot encoded) 
dummy_y = np_utils.to_categorical(encoded_Y) 
batch_size = 5
start_time = time.clock()
# define baseline model 


# create model
model = Sequential()
model.add(Dense(32, input_dim=4, kernel_initializer="normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(16, kernel_initializer="normal"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3, kernel_initializer="normal"))
model.add(Activation('softmax'))

# Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])


model.fit( X, dummy_y, epochs=200, batch_size=32, verbose=1) 
