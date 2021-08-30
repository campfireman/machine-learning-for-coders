import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])
model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

data = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
labels = np.array([-3.0, -1.0, 1.0, 3, 5.0, 7.0], dtype=float)

model.fit(data, labels, epochs=1000)

print(model.predict([10.0]))
print(f'Layer 0: {l0.get_weights()}')
