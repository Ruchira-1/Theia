from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
import numpy


X = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]])
y = numpy.array([[0.], [0.], [1.], [1.], [0.], [0.]])
model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(X, y, epochs=20)
