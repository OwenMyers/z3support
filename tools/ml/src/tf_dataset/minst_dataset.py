import tensorflow as tf
from keras.datasets import mnist


class MnistDataset:

    @staticmethod
    def __new__(cls, train=True, train_percent=80):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape(x_test.shape + (1,))

        #(x_train, y_train), (x_test, y_test) = load_mnist()
        x_train = x_train[:1000]
        x_test = x_test[:1000]
        if train:
            return x_train
        return x_test
        #(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        #train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        #if train:
        #    train_images = train_images[:int(len(train_images) * train_percent/100.0)]
        #else:
        #    train_images = train_images[:int(len(train_images) * (100 - train_percent) / 100.0)]
        #train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        #return train_images
