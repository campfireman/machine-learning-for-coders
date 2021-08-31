import urllib.request
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = 'I/3/data/validation/'


def download_validation_data():
    url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
    file_name = 'validation-horse-or-human.zip'
    urllib.request.urlretrieve(url, file_name)

    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(training_dir)


def get_validation_datagen():
    validation_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = validation_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )


if __name__ == '__main__':
    download_validation_data()
    validation_datagen = get_validation_datagen()
