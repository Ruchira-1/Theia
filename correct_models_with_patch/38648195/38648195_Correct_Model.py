from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, to_categorical
import time
import sys

# generate some data
dummyX, dummyY = make_classification(n_samples=4000, n_features=20, n_classes=3,n_informative=16,random_state=42)
dummyY = to_categorical(dummyY)



# neural network
model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
          optimizer='Adam',
          metrics=['accuracy'])


X_train, X_test, y_train, y_test = train_test_split(dummyX, dummyY, test_size=0.20)

model.fit(X_train, y_train,epochs=30, batch_size=30, validation_data=(X_test, y_test))



