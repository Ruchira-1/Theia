from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import numpy,pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression

numpy.random.seed(7)


X,Y = make_regression(n_samples=1000, n_features=2, n_informative=2,random_state=42)
print("Variables: \n", X)
print("Target_outputs: \n", Y)
sc = MinMaxScaler()
X = sc.fit_transform(X)
# create model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))


model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['MSE'])
# Fit the model
model.fit(X, Y, epochs=200, batch_size=32)


