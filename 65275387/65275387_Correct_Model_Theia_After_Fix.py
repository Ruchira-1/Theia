import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,  BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


X_train, y_train= make_classification(n_samples=1000,n_features=100*100*3,n_classes=7,n_informative=80*80*3,random_state=0)
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)


X_train = np.array(X_train).reshape(-1, 100, 100, 3)
y_train = np.array(y_train)
y_train = y_train.astype(int)
y_train = to_categorical(y_train)
# X_train = X_train/255.0



model = Sequential()
model.add(Conv2D(200, (3, 3), activation="relu", input_shape=(100, 100, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(200, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(200, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(200, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())        
model.add(Dense(142, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(7, activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

fittedModel = model.fit(X_train,y_train, epochs=30, verbose = 1)