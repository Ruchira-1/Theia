from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
batch_size =64
img_row,img_column = 24,24
epoch =15

X, y = make_classification(n_samples = 1000, n_features= 24*24*3, n_classes = 2, random_state=42,n_informative=20*20*3)

# sc = MinMaxScaler()
# X = sc.fit_transform(X)
X = X.reshape(-1,24,24,3)

# Create Model 
model = Sequential()

model.add(Conv2D(filters=64, input_shape=(img_row,img_column,3), kernel_size=(11,11), strides=(4,4), padding='same',))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same', ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', ))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))

opt=Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X,y,batch_size= batch_size, epochs=15, verbose =1)
