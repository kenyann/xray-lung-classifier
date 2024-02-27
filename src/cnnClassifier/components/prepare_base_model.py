from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, type):
        self.config = config
        self.type = type
        if self.type == 'f':
            self.params_image_size = config.params_model_f_image_size
            self.model_path = config.F_model_path
        if self.type == 'r':
            self.params_image_size = config.params_model_lr_image_size
            self.model_path = config.R_model_path
        if self.type == 'l':
            self.params_image_size = config.params_model_lr_image_size
            self.model_path = config.L_model_path

    def get_base_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(8, (5, 5), strides=1, padding='same',
                       activation='relu', input_shape=self.params_image_size))
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))

        self.model.add(Conv2D(8, (5, 5), strides=1,
                       padding='same', activation='relu'))
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))

        self.model.add(Conv2D(16, (5, 5), strides=1,
                       padding='same', activation='relu'))
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))

        self.model.add(Conv2D(16, (5, 5), strides=1,
                       padding='same', activation='relu'))
        self.model.add(MaxPool2D((2, 2), strides=2, padding='same'))

        self.model.add(Flatten())
        self.model.add(
            Dense(units=self.config.params_classes,  activation='softmax'))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"])

        self.save_model(path=self.model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
