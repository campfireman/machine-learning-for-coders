import os
import urllib

from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

from .training_data import get_train_datagen
from .validation_data import get_validation_datagen

weights_file = "data/inception_v3.h5"


def download():
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

    urllib.request.urlretrieve(weights_url, weights_file)


if not os.path.exists(weights_file):
    download()

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print(f'laster layer output shape: {last_layer.output_shape}')
last_output = last_layer.output

x = keras.layers.Flatten()(last_output)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(pre_trained_model.input, x)


def main():
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        get_train_datagen(target_size=(150, 150)),
        epochs=15,
        validation_data=get_validation_datagen(target_size=(150, 150))
    )

    model.save('data/horse-or-human/models')


if __name__ == '__main__':
    main()
