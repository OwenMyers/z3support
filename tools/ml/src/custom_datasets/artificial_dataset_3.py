import tensorflow as tf
import numpy as np


class ArtificialDataset3(tf.data.Dataset):

    def __new__(cls, train=True, train_percent=80):

        # Sensible results with 1000x scaling on the px_z term of loss
        #const = 2.0
        #individual_section_size = 1000
        #d1 = np.zeros([individual_section_size, 12, 12, 1]) - 1.0 + const
        #d2 = np.zeros([individual_section_size, 12, 12, 1]) + 1.0 + const
        #np.random.seed(1)
        #d3 = np.random.randint(2, size=(individual_section_size, 12, 12, 1)) * 2 + int(const - 1)

        const = 0.0
        individual_section_size = 1000
        d1 = np.zeros([individual_section_size, 12, 12, 1]) - 1.0 + const
        d2 = np.zeros([individual_section_size, 12, 12, 1]) + 1.0 + const
        np.random.seed(1)
        d3 = np.random.randint(2, size=(individual_section_size, 12, 12, 1)) * 2 + int(const - 1)

        final = np.append(d1, d2, axis=0)
        final = np.append(final, d3, axis=0)

        labels = np.zeros(individual_section_size) + 1.0
        labels = np.append(labels, np.zeros(individual_section_size) + 2.0)
        labels = np.append(labels, np.zeros(individual_section_size) + 3.0)

        zipped = list(zip(final, labels))
        np.random.shuffle(zipped)
        final_o, labels_o = zip(*zipped)

        n_records = len(final_o)

        x_train = final_o[: int(n_records * train_percent/100.0)]
        x_test = final_o[int(n_records * train_percent/100.0):]
        y_train = labels_o[: int(n_records * train_percent/100.0)]
        y_test = labels_o[int(n_records * train_percent/100.0):]
        if train:
            return np.array(x_train).astype('float32'), y_train
        return np.array(x_test).astype('float32'), y_test