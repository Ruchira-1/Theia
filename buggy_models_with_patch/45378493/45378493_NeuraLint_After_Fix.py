
import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification



X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
print(X.shape)
X = X.reshape(X.shape[0],X.shape[1],  1)


print('Shape of X',X.shape)
model = Sequential()
model.add(Conv1D(input_shape = (4,1),
                       filters=16,
                        kernel_size=4,
                       padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Conv1D(filters=8,
                        kernel_size=4,
                        padding='same'))
# model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64))
# model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, y, 
          epochs = 100, 
          batch_size = 128, 
          verbose=1, 
          shuffle=True)
     
