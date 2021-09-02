import csv
import os

import numpy as np
import tensorflow_datasets as tfds
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .conv_trans_horses_humans import model

EPOCHS = 25
TEST_DIR = 'data/cat-or-dog/test'
TRAINING_DIR = 'data/cat-or-dog/training/'
MODEL_PATH = f'data/cat-or-dog/models/{EPOCHS}-epochs.h5'

target_size = (150, 150)


def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    train_datagen = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=target_size,
        class_mode='binary'
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        train_datagen,
        epochs=EPOCHS,
    )

    model.save(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    train_model()

model = keras.models.load_model(MODEL_PATH)

with open('data/cat-or-dog/result.csv', 'w') as file:
    filewriter = csv.writer(file, delimiter=',')
    filewriter.writerow(['id', 'label'])
    for file in os.listdir(TEST_DIR):
        idx = file.split('.')[0]
        filepath = os.path.join(TEST_DIR, file)
        image = keras.preprocessing.image.load_img(
            filepath, target_size=target_size)
        image = keras.preprocessing.image.img_to_array(image) / 255
        image = np.array([image])
        print(filepath)
        prediction = model.predict(image)[0][0]
        filewriter.writerow([idx, prediction])
        print(f'id: {idx} predicted: {prediction}')
