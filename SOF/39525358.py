

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import numpy as np
import time
import sys
import keras 
from sklearn.preprocessing import StandardScaler


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
batch_size = 10
#dataset = numpy.loadtxt("sorted output.csv", delimiter=",")
dataset =  np.loadtxt('sorted output.csv', delimiter=',', skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[:,0:3]
Y = dataset[:,3]
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# create model

model = Sequential()
model.add(Dense(12, input_dim=3, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(3, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(1, init='uniform'))
model.add(Activation('sigmoid'))
# Compile model
start_time = time.clock()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# Fit the model
number = len(model.layers)
model.fit(X_train, y_train,  nb_epoch=150, batch_size=1, verbose =2)
end_time = time.clock()

print("Time =",(end_time -start_time))
sys.exit(1)