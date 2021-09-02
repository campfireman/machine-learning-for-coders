import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

TARGET_SIZE = (150, 150)
TRAINING_DIR = 'data/rock-paper-scissors/train'
VALIDATION_DIR = 'data/rock-paper-scissors/train'
TEST_DIR = 'data/rock-paper-scissors/test'
MODEL_PATH = 'data/rock-paper-scissors/25-epochs.h5'


def train_model():
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    training_datagen = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=TARGET_SIZE,
        class_mode='categorical',
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
    )
    validation_datagen = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=TARGET_SIZE,
        class_mode='categorical',
    )

    model = keras.models.Sequential([
        # Note the input shape is the desired size of the image:
        # 150x150 with 3 bytes color
        # This is the first convolution
        keras.layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        keras.layers.Flatten(),
        # 512 neuron hidden layer
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(training_datagen, epochs=25,
                        validation_data=validation_datagen, verbose=1)

    model.save(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    train_model()

model = keras.models.load_model(MODEL_PATH)

for file in os.listdir(TEST_DIR):
    idx = file.split('.')[0]
    filepath = os.path.join(TEST_DIR, file)
    image = keras.preprocessing.image.load_img(
        filepath, target_size=TARGET_SIZE)
    image = keras.preprocessing.image.img_to_array(image) / 255
    image = np.array([image])
    prediction = model.predict(image)
    print(f'name: {idx} predicted: {prediction}')
