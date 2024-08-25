from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

batch_size = 32
epoch = 30
img_rows, img_col = 32,32
LEARNING_RATE = 0.001


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_test = x_test.reshape(x_test.shape[0],32,32,3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

cnn = Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(32, 32, 3)))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.2))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(50, (3, 3),activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.2))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Conv2D(100, (3, 3),activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.2))
cnn.add(layers.MaxPooling2D((2, 2)))
cnn.add(layers.Flatten())
cnn.add(layers.Dense(100, activation='relu'))
cnn.add(layers.BatchNormalization())
cnn.add(layers.Dropout(0.2))
cnn.add(layers.Dense(10, activation='softmax'))



cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


history_cnn = cnn.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
                validation_data=(x_test, y_test),verbose=1)
              