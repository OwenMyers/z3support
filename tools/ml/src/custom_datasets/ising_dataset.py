import numpy as np
import logging
import os


class IsingDataset:

    @staticmethod
    def __new__(cls, train=True, train_percent=80, path_list=None, lattice_size=None, configuration_shape='links'):
        """
        The `path_list` will contain a list of the paths to different binary data files. There SHOULD (should is in caps
        because you can use this with mixed data, but only do so if the mixed data is from the same system,
        perhaps different temperatures for example, and if it is already balanced) be one for each of the
        "different types of systems" (e.g. z3, z2, high temp, etc). There is no guarantee that there are the same
        number of lattice configurations in each file, but the `import_data, function below takes care of all of
        that for us to make sure we get a balanced dataset and that it meets some basic requirements.

        :param train: Are you getting the training or testing set
        :param train_percent: Percentage of data to be used for training. What remains is the testing data set. No
            functionality for a hold-out set yet.
        :param path_list: List of the different datasets (See primary description above)
        """
        if path_list is not None:
            raise ValueError("Setting path list in init so don't set outside")
        if lattice_size is not None:
            raise ValueError("Setting lattice size in init so don't set outside")

        lattice_size = 40
        path_list = [
            #"/home/owen/repos/z3support/data/split_ising_data/t1.npy",
            "/home/owen/repos/z3support/data/split_ising_data/t2.npy",
        ]
        all_data, data_labels = cls.import_data(path_list)

        n_records = len(all_data)
        x_train = all_data[: int(n_records * train_percent/100.0)]
        x_test = all_data[int(n_records * train_percent/100.0):]
        x_train = np.reshape(x_train, (len(x_train), lattice_size, lattice_size, 1))
        x_test = np.reshape(x_test, (len(x_test), lattice_size, lattice_size, 1))

        y_train = data_labels[: int(n_records * train_percent/100.0)]
        y_test = data_labels[int(n_records * train_percent/100.0):]

        #x_train, y_train = cls.augment_data(x_train, y_train)
        #x_test, y_test = cls.augment_data(x_test, y_test)

        if train:
            return x_train, y_train
        return x_test, y_test

    @staticmethod
    def augment_data(x, y):
        to_add_x_list = []
        to_add_y_list = []
        for cur_x, cur_y in zip(x,y):
            to_add_x_list.append(np.fliplr(cur_x))
            to_add_y_list.append(cur_y)
            to_add_x_list.append(np.rot90(cur_x))
            to_add_y_list.append(cur_y)

        return np.append(x, to_add_x_list, axis=0), np.append(y, to_add_y_list, axis=0)


    @staticmethod
    def import_data(list_data):
        """
        :param list_data: List of ``.npy`` files (full paths).
        Returns:
            Array of all configurations. numpy, float32 as first element. mean spin as the second.
        """
        logging.info("Starting data import")
        logging.info("    Loading data")
        # * Load all data into memory
        loaded_data_list = []
        length_list = []
        for current_data_path in list_data:
            logging.debug(f"        Current path to data being loaded {current_data_path}")
            current_loaded = np.load(current_data_path)
            loaded_data_list.append(current_loaded)
            length_list.append(len(current_loaded))
        logging.info("    Data loaded")

        data_labels = []
        for i, current_data in enumerate(loaded_data_list):
            for cur_row in current_data:
                data_labels.append(np.mean(cur_row))
        concatenated = np.vstack(loaded_data_list)

        data_set_with_indices = list(zip(concatenated, data_labels))
        # * Scramble but keep original labels in separate list
        np.random.seed(1)
        np.random.shuffle(data_set_with_indices)
        # * Return both the scrambled data set and the separate list... separately.
        concatenated, data_labels = zip(*data_set_with_indices)
        return np.array(concatenated).astype('float32'), data_labels