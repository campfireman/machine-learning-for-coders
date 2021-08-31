import tensorflow as tf
import tensorflow.keras as keras
from I.utils import TrainingTargetCallback

data = keras.datasets.fashion_mnist
callbacks = TrainingTargetCallback()
callbacks.training_target = 0.93

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)
