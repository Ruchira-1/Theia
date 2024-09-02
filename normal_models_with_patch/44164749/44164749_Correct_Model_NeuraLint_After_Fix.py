
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from sklearn.datasets import make_multilabel_classification


X, y = make_multilabel_classification(n_samples = 10000, n_classes = 11,n_labels=3, random_state=42)

model = Sequential()
model.add(Dense(5000, input_dim=X.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(600))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
start_time = time.time()

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy',])

model.fit(X,y,epochs=10,batch_size=2000)