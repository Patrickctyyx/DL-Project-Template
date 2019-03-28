# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from bases.model_base import ModelBase


class Vgg16Model(ModelBase):

    def __init__(self, config, model_path=None):
        super(Vgg16Model, self).__init__(config)
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):

        main_input = tf.keras.Input(shape=(224, 224, 3), name='input')

        conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv1_1')(main_input)
        conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv1_2')(conv1_1)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_2)

        conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv2_1')(pool1)
        conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv2_2')(conv2_1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_2)

        conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv3_1')(pool2)
        conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv3_2')(conv3_1)
        conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv3_3')(conv3_2)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_3)

        conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv4_1')(pool3)
        conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv4_2')(conv4_1)
        conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv4_3')(conv4_2)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_3)

        conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv5_1')(pool4)
        conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv5_2')(conv5_1)
        conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu', kernel_initializer='uniform',
                                         name='conv5_3')(conv5_2)
        pool5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(conv5_3)

        fl_6 = tf.keras.layers.Flatten()(pool5)
        fc_6 = tf.keras.layers.Dense(4096, activation='relu', name='fc6')(fl_6)
        drop_6 = tf.keras.layers.Dropout(0.5)(fc_6)

        fc_7 = tf.keras.layers.Dense(4096, activation='relu', name='fc7')(drop_6)
        drop_7 = tf.keras.layers.Dropout(0.5)(fc_7)

        fc_8 = tf.keras.layers.Dense(1000, activation='softmax')(drop_7)

        model = tf.keras.Model(inputs=main_input, outputs=fc_8)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd', metrics=['accuracy']
        )

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, 'vgg16.png'), show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)
