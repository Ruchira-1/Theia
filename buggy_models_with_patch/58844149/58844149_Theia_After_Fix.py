from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd 
from keras.layers import Dropout
from sklearn.model_selection import train_test_split


batch_size =64
img_row,img_column = 28,28
epoch =20
train_path = '../dogs-vs-cats/train/'
test_path = '../dogs-vs-cats/test1/'
#train = pd.DataFrame({'file': os.listdir('dogs-vs-cats/train/')})
train = pd.DataFrame({'file': os.listdir(train_path)})
#file_names = os.listdir(train_path)
labels = []
binary_labels = []
for path in os.listdir(train_path):
   if 'dog' in path:
        labels.append('dog')
        binary_labels.append(1)
   else:
        labels.append('cat')
        binary_labels.append(0)

train['labels'] = labels
train['binary_labels'] = binary_labels
print(train.head())
test = pd.DataFrame({'file': os.listdir(test_path)})
train_set, val_set = train_test_split(train,
                                     test_size=0.2,random_state=42)
print(len(train_set), len(val_set))

datagen_train = ImageDataGenerator(rescale=1./255,
                          shear_range=0.2, zoom_range=0.2,
                           horizontal_flip=True)

# Augment validating data
datagen_valid = ImageDataGenerator(rescale=1./255)

aug_train = datagen_train.flow_from_dataframe(train_set, directory=train_path, 
                                             x_col='file', y_col='labels',
                                             target_size=(28,28), class_mode='binary',
                                             batch_size=batch_size)

aug_test = datagen_valid.flow_from_dataframe(val_set, directory=train_path,
                                             x_col='file', y_col='labels',
                                             target_size=(28,28), class_mode='binary',
                                             batch_size=batch_size)


model = Sequential()

model.add(Conv2D(96, (11,11), padding="same", input_shape=(28,28, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(128, (3,3), padding="same", activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3,3), padding="same", activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())
# model.add(Dense(units=1000, activation='relu'  ))
model.add(Dense(units= 300, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(aug_train, epochs=epoch, verbose=1)
