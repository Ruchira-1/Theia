import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import time
import sys
import keras
from sklearn.preprocessing import MinMaxScaler

batch_size = 32
epochs = 20
alpha = 0.0001
lambda_ = 0
h1 = 50

train = pd.read_csv('../mnist_train.csv.zip')
test = pd.read_csv('../mnist_test.csv.zip')

train = train.loc['1':'5000', :]
test = test.loc['1':'2000', :]

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

x_train = train.loc[:, '1x1':'28x28']
y_train = train.loc[:, 'label']

x_test = test.loc[:, '1x1':'28x28']
y_test = test.loc[:, 'label']

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

nb_classes = 10
targets = y_train.reshape(-1)
y_train_onehot = np.eye(nb_classes)[targets]

nb_classes = 10
targets = y_test.reshape(-1)
y_test_onehot = np.eye(nb_classes)[targets]

model = keras.Sequential()
model.add(keras.layers.Dense(784, input_shape=(784,)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(h1,  kernel_regularizer=tf.keras.regularizers.l2(lambda_)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(lambda_)))
model.add(keras.layers.Activation('softmax'))

model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam',
             metrics = ['accuracy'])

model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size)
       
