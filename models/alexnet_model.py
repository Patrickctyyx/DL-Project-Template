# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from bases.model_base import ModelBase


class AlexNetModel(ModelBase):

    def __init__(self, config, model_path=None):
        super(AlexNetModel, self).__init__(config)
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):
        main_input = tf.keras.Input(shape=(227, 227, 3), name='input')

        conv_1 = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid',
                                        activation='relu', kernel_initializer='uniform',
                                        name='conv1')(main_input)
        pool_1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(conv_1)

        conv_2 = tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                                        activation='relu', kernel_initializer='uniform',
                                        name='conv2')(pool_1)
        pool_2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(conv_2)

        conv_3 = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                                        activation='relu', kernel_initializer='uniform',
                                        name='conv3')(pool_2)

        conv_4 = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                                        activation='relu', kernel_initializer='uniform',
                                        name='conv4')(conv_3)

        conv_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                        activation='relu', kernel_initializer='uniform',
                                        name='conv5')(conv_4)
        pool_5 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(conv_5)

        fl_6 = tf.keras.layers.Flatten()(pool_5)
        fc_6 = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(fl_6)
        drop_6 = tf.keras.layers.Dropout(0.5)(fc_6)

        fc_7 = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(drop_6)
        drop_7 = tf.keras.layers.Dropout(0.5)(fc_7)

        fc_8 = tf.keras.layers.Dense(1000, activation='softmax')(drop_7)

        model = tf.keras.Model(inputs=main_input, outputs=fc_8)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd', metrics=['accuracy']
        )

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, 'alexnet.png'), show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)
