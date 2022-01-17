import tensorflow as tf
from tensorflow.keras.datasets import mnist


class MnistDataset:

    @staticmethod
    def __new__(cls, train=True, train_percent=80):
        """Just a wrapper for MNIST dataset which is convenient for making the encoder dataset "stuff" more general"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape(x_test.shape + (1,))

        if train:
            return x_train[:4000], y_train[:4000]
        return x_test[:4000], y_test[:4000]

#        if train:
#            return x_train, y_train
#        return x_test, y_test
