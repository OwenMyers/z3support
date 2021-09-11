import tensorflow as tf


class MnistDataset(tf.data.Dataset):
    def _generator(cls, batch_size, train=True, train_percent=80):
        print("--------------------------------------------------- IN MNIST GENERATOR -------------------------------")
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

        if train:
            train_images = train_images[:int(len(train_images) * train_percent/100.0)]
        else:
            train_images = train_images[:int(len(train_images) * (100 - train_percent) / 100.0)]
        #train_images = tf.expand_dims(train_images, axis=0)
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        if train:
            num_samples = int(len(train_images) * train_percent / 100.0)
        else:
            num_samples = int(len(train_images) * (100 - train_percent) / 100.0)

        for sample_idx in range(num_samples):
            yield (train_images[sample_idx], train_images[sample_idx])
        #train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
        #train_dataset = tf.expand_dims(train_dataset, axis=0)

    def __new__(cls, batch_size, train=True, train_percent=80):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32), tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32)),
            args=(0, train, batch_size)
        )
