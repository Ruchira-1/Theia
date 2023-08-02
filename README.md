# Theia
To use Theia, you need to add our callback as a subclass in your keras.callbacks.py file.

Then you can pass our callback Theia() to the .fit() method for the Keras model as follows:

```
callback = keras.callbacks.Theia(train_inputs, test_inputs, batch_size, problem_type,input_type)
model = keras.models.Sequential()
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.compile(keras.optimizers.SGD(), loss='mse')
model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=50, 
...                     callbacks=[callback], verbose=0)
```
For PyTorch programs call Theia() before the training loop:

```
Theia.check(train_data, test_data, model, loss_fun, optimizer, batch_size, problem_type,input_type)
```

This repository contains:
[Benchmark -- 40 buggy Stack Overflow Programs] (https://github.com/anoau/Theia/tree/main/SOF)
