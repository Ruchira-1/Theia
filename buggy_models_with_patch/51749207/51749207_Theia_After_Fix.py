
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=1000, n_features=600, n_classes=2, random_state=42)
print(X.shape)
sc = MinMaxScaler()
X = sc.fit_transform(X)

X = X.reshape(X.shape[0], X.shape[1], 1)


model = Sequential()
model.add(Conv1D(filters=20, kernel_size=4,padding='same',input_shape=(600,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(50,  input_dim = 600))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=['accuracy'])

model.fit(X,y, epochs = 100, batch_size=32, verbose=1)

