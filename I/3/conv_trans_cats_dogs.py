from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .conv_trans_horses_humans import model

test_dir = 'data/cat-or-dog/test/'
training_dir = 'data/cat-or-dog/training/'
target_size = (150, 150)
test_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    class_mode='binary'
)

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
    training_dir,
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
    epochs=20,
)

model.save('data/cat-or-dog/models/20-epochs.h5')
