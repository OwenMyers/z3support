import numpy as np
import logging
import os


class PhysicsDataset:

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
        :param lattice_size: Edge length of lattice
        """

        all_data, data_labels = cls.import_data(path_list)

        n_records = len(all_data)
        x_train = all_data[: int(n_records * train_percent/100.0)]
        x_test = all_data[int(n_records * train_percent/100.0):]
        if configuration_shape == 'links':
            x_train = np.reshape(x_train, (len(x_train), lattice_size * 2, lattice_size * 2, 1))
            x_test = np.reshape(x_test, (len(x_test), lattice_size * 2, lattice_size * 2, 1))
        elif configuration_shape == 'plain':
            x_train = np.reshape(x_train, (len(x_train), lattice_size, lattice_size, 1))
            x_test = np.reshape(x_test, (len(x_test), lattice_size, lattice_size, 1))
        else:
            raise ValueError(f"No way to handle shape specification provided by user: {configuration_shape}")

        y_train = data_labels[: int(n_records * train_percent/100.0)]
        y_test = data_labels[int(n_records * train_percent/100.0):]

        if train:
            return x_train, y_train
        return x_test, y_test

    @staticmethod
    def import_data(list_data):
        """
        Get all the data from different systems into one place to be passed into the machine learning algorithm.

        (We will want to expand and test this if we have time or if we start doing more than just the z2 and z3 data)

        :param list_data: List of ``.npy`` files (full paths).

        This function will load ALL of the data into memory. If the data sets get too large this will be a breaking
        point.

        Steps:
            * Load all data into memory
            * Find what the maximum and minimum values of each are. Check equivalency
            * Normalize all data (divide by max)
            * Balance the data sets (under sample to the lowest number of configurations)
            * Put together into a single entity
            * Scramble but keep original labels in separate list
            * Return both the scrambled data set and the separate list... separately.

        Returns:
            Array of all configurations. numpy, float32 as first element. Original indices of the data as the second.
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
        # * Find what the maximum and minimum values of each are. Check equivalency
        max_list = []
        min_list = []
        for current_dataset in loaded_data_list:
            max_list.append(np.max(current_dataset))
            min_list.append(np.min(current_dataset))
        logging.debug(f"    Max values in data: {max_list}")
        logging.debug(f"    Min values in data: {min_list}")
        for current_max in max_list:
            for second_max in max_list:
                if current_max != second_max:
                    raise ValueError("Different maximums")
        for current_min in min_list:
            for second_min in min_list:
                if current_min != second_min:
                    raise ValueError("Different minimums")

        # * Normalize all data (divide by max)
        normalized_data_list = []
        for current_dataset in loaded_data_list:
            normalized_data_list.append(current_dataset / max_list[0])

        # * Balance the data sets (under sample to the lowest number of configurations)
        # At the end of this the balanced_dataset will be a list of the number of full data sets that have been passed
        # in. These still need to be put together as a single entity.
        balanced_dataset = []
        for current_dataset in normalized_data_list:
            if len(current_dataset) > min(length_list):
                random_indices = np.random.choice(current_dataset.shape[0], min(length_list), replace=False)
                balanced_dataset.append(current_dataset[random_indices, :])
            else:
                balanced_dataset.append(current_dataset)

        # * Put together into a single entity
        data_labels = []
        for i, current_data_path in enumerate(list_data):
            current_label = os.path.basename(os.path.normpath(current_data_path))
            data_labels += [current_label] * len(balanced_dataset[0])
        concatenated = np.vstack(balanced_dataset)

        data_set_with_indices = list(zip(concatenated, data_labels))
        # * Scramble but keep original labels in separate list
        np.random.shuffle(data_set_with_indices)
        # * Return both the scrambled data set and the separate list... separately.
        concatenated, data_labels = zip(*data_set_with_indices)
        return np.array(concatenated).astype('float32'), data_labels