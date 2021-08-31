from tensorflow import keras


class TrainingTargetCallback(keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_target = 0.98

    @property
    def training_target(self):
        return self._training_target

    @training_target.setter
    def training_target(self, new_target):
        self._training_target = new_target

    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > self.training_target):
            print(f'\nReached {self.training_target} stopping')
            self.model.stop_training = True
