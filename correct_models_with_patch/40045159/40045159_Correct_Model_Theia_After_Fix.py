from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
import os
import pandas as pd 
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
                                             target_size=(28,28), class_mode='categorical',
                                             batch_size=batch_size)

aug_test = datagen_valid.flow_from_dataframe(val_set, directory=train_path,
                                             x_col='file', y_col='labels',
                                             target_size=(28,28), class_mode='categorical',
                                             batch_size=batch_size)


model = Sequential()
model.add(Conv2D(32, (3,3), padding='valid', input_shape=(28, 28,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))


sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(aug_train, validation_data=aug_test, epochs=epoch, verbose=1)