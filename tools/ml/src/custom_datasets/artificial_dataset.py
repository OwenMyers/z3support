import tensorflow as tf
import time


class ArtificialDataset(tf.data.Dataset):

    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_shapes=(1,),
            output_types=tf.int64,
            args=(num_samples,)
        )