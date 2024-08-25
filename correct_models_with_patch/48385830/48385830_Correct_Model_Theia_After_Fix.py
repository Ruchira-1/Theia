import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.initializers import RandomNormal

batch_size = 10
# import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
input_dim = (img_rows * img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)

# Construct model
# 784 * 30 * 10
# Normal distribution for weights/biases
# Stochastic Gradient Descent optimizer
# Mean squared error loss (cost function)
model = Sequential()
model.add(Dense(30,
               input_dim=input_dim,
               activation='relu',
               kernel_initializer=RandomNormal(stddev=1),
               bias_initializer=RandomNormal(stddev=1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
#model.add(layer1)
model.add(Dense(10,
                activation='softmax',
               kernel_initializer=RandomNormal(stddev=1),
               bias_initializer=RandomNormal(stddev=1)))
#model.add(layer2)
# print('Layer 1 input shape: ', layer1.input_shape)
# print('Layer 1 output shape: ', layer1.output_shape)
# print('Layer 2 input shape: ', layer2.input_shape)
# print('Layer 2 output shape: ', layer2.output_shape)

model.summary()
start = time.clock() 
model.compile(optimizer=SGD(lr=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Train 
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=30,
          verbose=1)
