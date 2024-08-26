from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import os

batch_size =64
img_row,img_column = 150,150
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
                                             target_size=(150,150), class_mode='categorical',
                                             batch_size=batch_size)

aug_test = datagen_valid.flow_from_dataframe(val_set, directory=train_path,
                                             x_col='file', y_col='labels',
                                             target_size=(150,150), class_mode='categorical',
                                             batch_size=batch_size)


model = Sequential()

model.add(Conv2D(32, 3, activation='relu', strides=(1,1), padding='same', input_shape=(150,150,3), ))
model.add(Conv2D(32, 3, activation='relu',strides=(2,2), padding='same'))
model.add(Conv2D(64, 3, activation='relu', strides=(2,2), padding='same'))
model.add(Conv2D(128, 3, activation='relu', strides=(2,2), padding='same'))
model.add(Conv2D(256, 3, activation='relu', strides=(2,2), padding='same'))
model.add(Flatten())
model.add(Dense(units= 512, activation='relu'))
model.add(Dense(2))
model.add(Activation("sigmoid"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(aug_train, validation_data=aug_test, epochs=epoch, verbose=1,)