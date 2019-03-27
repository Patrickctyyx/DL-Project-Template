# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from bases.model_base import ModelBase


class MnistModel(ModelBase):

    def __init__(self, config, model_path=None):
        super(MnistModel, self).__init__(config)
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):
        # the input vector is 1d
        main_input = tf.keras.Input(shape=(28, 28, 1), name='mnist_input')
        
        layer_1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv1')(main_input)
        layer_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(layer_1)
        layer_1 = tf.keras.layers.BatchNormalization()(layer_1)
        layer_1 = tf.keras.layers.Activation('relu')(layer_1)
        # layer_1 = tf.keras.layers.Dense(32, activation='relu')(main_input)

        layer_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2')(layer_1)
        layer_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(layer_2)
        layer_2 = tf.keras.layers.BatchNormalization()(layer_2)
        layer_2 = tf.keras.layers.Activation('relu')(layer_2)
        # layer_2 = tf.keras.layers.Dense(64, activation='relu')(layer_1)

        # layer_2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', name='conv3')(layer_2)
        # layer_2 = tf.keras.layers.Activation('relu')(layer_2)

        out = tf.keras.layers.Flatten()(layer_2)
        out = tf.keras.layers.Dense(10, activation='softmax')(out)

        model = tf.keras.Model(inputs=main_input, outputs=out)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=self.config.lr),
            metrics=['accuracy']
        )

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, "mnist_model.png"), show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)
