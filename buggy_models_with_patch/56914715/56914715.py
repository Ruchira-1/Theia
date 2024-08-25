import os

import random
import numpy as np
from numpy.lib.stride_tricks import as_strided
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical



X, y = make_classification(n_samples = 1000, n_features= 25*25, n_classes = 2, random_state=42,n_informative=20*20)

sc = MinMaxScaler()
X = sc.fit_transform(X)
X = X.reshape(-1,25,25,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = Sequential()
model.add(Conv2D(128, (3,3), input_shape = (25,25,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("softmax"))

model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ['accuracy'])

model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs = 15, verbose=1,batch_size =64)


    
