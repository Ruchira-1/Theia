import numpy as np
#import matplotlib.pyplot as plt 
from tensorflow import keras
import os
import random 
# import cv2
import random 
from tensorflow.keras.optimizers import SGD
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, y_train= make_classification(n_samples=1000,n_features=50*50*3,n_classes=16,n_informative=20*20*3,random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)


X_train = np.array(X_train).reshape(-1, 50, 50, 3)
y_train = np.array(y_train)
y_train = y_train.astype(int)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(16))
model.add(Activation('softmax'))



model.compile(loss='sparse_categorical_crossentropy', 
             optimizer=SGD(learning_rate=0.01), 
             metrics=['accuracy'])
model_fit = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1)
