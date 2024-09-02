
import tensorflow
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_regression


X, y = make_regression(n_samples=5, n_features=600, noise=1, random_state=42)
print(X.shape)
X = X.reshape(X.shape[0], X.shape[1], 1)


model = Sequential()
model.add(Conv1D(filters=20, kernel_size=4,padding='same',input_shape=(600,1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(50,  input_dim = 600))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=['accuracy'])

model.fit(X,y, epochs = 100, batch_size=8, verbose=1)

