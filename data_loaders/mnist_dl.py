# -*- coding: utf-8 -*-

import tensorflow as tf 

from bases.data_loader_base import DataLoaderBase


class MnistDL(DataLoaderBase):
    def __init__(self, config=None):
        super(MnistDL, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        self.X_train = self.X_train.reshape((-1, 28, 28, 1))
        self.X_test = self.X_test.reshape((-1, 28, 28, 1))

        self.y_train = tf.keras.utils.to_categorical(self.y_train)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)

        print("[INFO] X_train.shape: %s, y_train.shape: %s" \
              % (str(self.X_train.shape), str(self.y_train.shape)))
        print("[INFO] X_test.shape: %s, y_test.shape: %s" \
              % (str(self.X_test.shape), str(self.y_test.shape)))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
