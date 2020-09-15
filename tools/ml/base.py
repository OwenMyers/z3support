import os
import configparser
import logging
import numpy as np
from keras.callbacks import ModelCheckpoint
import keras
from tensorboard.plugins.hparams import api as hp


class MLToolMixin:

    def __init__(self, settings_file, working_location):

        # Make sure the required subdirectories are present
        assert os.path.exists(os.path.join(working_location, 'model_checkpoints'))
        assert os.path.exists(os.path.join(working_location, 'models'))
        assert os.path.exists(os.path.join(working_location, 'settings'))
        assert os.path.exists(os.path.join(working_location, 'tensorboard_raw'))
        assert os.path.exists(os.path.join(working_location, 'study_data'))

        if not os.path.exists(settings_file):
            raise ValueError(f"Can't find specified settings file {settings_file}")
        config = configparser.ConfigParser()
        config.read(settings_file)

        self.timestamp = config['Settings']['timestamp']
        self.L = int(config['Settings']['L'])
        self.feature_map_start = int(config['Settings']['FEATURE_MAP_START'])
        self.epochs = int(config['Settings']['EPOCHS'])
        # quick run of single param or full param sweep. Use True for testing.
        self.quick_run = config['Settings']['QUICK_RUN']
        self.verbose = config['Settings']['VERBOSE']
        self.tensorboard_sub_dir = config['Settings']['TENSORBOARD_SUB_DIR']
        self.checkpoint_file = os.path.join(working_location, 'model_checkpoints',
                                            'checkpoint_{}.hdf5'.format(config['Settings']['timestamp']))
        self.best_model_file = os.path.join(working_location, 'models',
                                            'best_hyper_param_autoencoder_{}'.format(config['Settings']['timestamp']))
        self.best_activations_file = os.path.join(
            working_location,
            'models',
            'best_hyper_param_activations_{}'.format(config['Settings']['timestamp'])
        )
        self.study_data_location = os.path.join(
            working_location,
            'study_data',
            config['Settings']['timestamp']
        )
        self.training_data_location = os.path.join(self.study_data_location, 'training_data.npy')
        self.testing_data_location = os.path.join(self.study_data_location, 'testing_data.npy')
        self.data_label_location = os.path.join(self.study_data_location, 'data_labels.npy')
        if not os.path.exists(self.study_data_location):
            os.mkdir(self.study_data_location)
        # This will be a list of the different sources, e.g. path to transformed Z3 data, and path to transformed Z2
        # data.
        self.data = self.get_all_data_sources(config)
        self.checkpointer = ModelCheckpoint(
            filepath=self.checkpoint_file,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
        self.run_location = working_location

        self.hp_batch_size = hp.HParam('batch_size', hp.Discrete([50]))
        self.hp_n_layers = hp.HParam('n_layers', hp.Discrete([3]))
        self.hp_feature_map_step = hp.HParam('feature_map_step', hp.Discrete([16]))
        self.hp_stride_size = hp.HParam('stride', hp.Discrete([1]))
        self.tensorboard_sub_dir = 'quick_run'
        if not self.quick_run:
            self.hp_batch_size = hp.HParam('batch_size', hp.Discrete([15, 50]))
            self.hp_n_layers = hp.HParam('n_layers', hp.Discrete([2, 3]))
            self.hp_feature_map_step = hp.HParam('feature_map_step', hp.Discrete([2, 8, 16]))
            self.hp_stride_size = hp.HParam('stride', hp.Discrete([1, 2]))

    def get_best_autoencoder(self):
        return keras.models.load_model(self.best_model_file)

    def get_best_activations(self):
        return keras.models.load_model(self.best_activations_file)

    @staticmethod
    def import_data(list_data):
        """
        Get all the data from different systems into one place to be passed into the machine learning algorithm.

        (We will want to expand and test this if we have time or if we start doing more than just the z2 and z3 data)

        Arguments:
            Requires list of ``.npy`` files.

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

    @staticmethod
    def get_all_data_sources(settings_file_parser):
        """
        Arguments:
            settings_file_parser (ConfigParser): The config parser instance containing ``DATA1, DATA2, ..., DATA<N>``

        Returns:
            A list of the full paths to all ``.npy`` files. Remember that each file contains the full list of
            configurations for that type (e.g. Z2, Z3, High temp, etc) transformed and ready for the neural network.
        """
        path_list = []
        got_all = False
        c = 1
        current_data_path = settings_file_parser['Data'][f'DATA{c}']
        while not got_all:
            if not os.path.exists(current_data_path):
                raise ValueError(f'Data path {current_data_path} does not exist.')
            path_list.append(current_data_path)
            try:
                c += 1
                current_data_path = settings_file_parser['Data'][f'DATA{c}']
            except KeyError:
                got_all = True

        return path_list
