import urllib.request
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = 'data/horse-or-human/training/'


def download_training_data():
    file_name = 'horse-or-human.zip'
    url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
    urllib.request.urlretrieve(url, file_name)

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(training_dir)


def get_train_datagen(target_size=(300, 300)):
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

    return train_datagen.flow_from_directory(
        training_dir,
        target_size=target_size,
        class_mode='binary'
    )


if __name__ == '__main__':
    download_training_data()
    train_datagen = get_train_datagen()
