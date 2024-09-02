from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
import pandas as pd
import numpy as np 
from sklearn import preprocessing


np.random.seed(1919)

### Constants ###

batch_size = 4
nb_epoch = 20

### load train and test ###
train  = pd.read_csv('../train.csv', index_col=0)
test  = pd.read_csv('../test.csv', index_col=0)
print ("Data Read complete")

Y = train.Survived
from keras.utils import to_categorical
en = preprocessing.LabelEncoder()
Y = en.fit_transform(Y)

print('This is Y',Y[:5])
train.drop('Survived', axis=1, inplace=True)

columns = train.columns

test_ind = test.index

train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# category_index = [0,1,2,3,4,5,6,8,9]
category_index = len(columns)
print('Length of columns',category_index)
for i in range(category_index):
    print (str(i)+" : "+columns[i])
    train[columns[i]] = train[columns[i]].fillna('missing')
    test[columns[i]] = test[columns[i]].fillna('missing')

train = np.array(train)
test = np.array(test)

### label encode the categorical variables ###
for i in range(category_index):
    print (str(i)+" : "+str(columns[i]))
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])

### making data as numpy float ###
train = train.astype(np.float32)
test = test.astype(np.float32)
#Y = np.array(Y).astype(np.int32)

model = Sequential()
model.add(Dense(512,input_dim =len(columns)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, input_dim =512))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(train, Y, epochs=nb_epoch, batch_size=batch_size, validation_split=0.20)
