import urllib.request
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = 'I/3/data/training/'


def download_training_data():
    file_name = 'horse-or-human.zip'
    url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
    urllib.request.urlretrieve(url, file_name)

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(training_dir)


def get_train_datagen():
    train_datagen = ImageDataGenerator(rescale=1/255)

    return train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )


if __name__ == '__main__':
    download_training_data()
    train_datagen = get_train_datagen()
