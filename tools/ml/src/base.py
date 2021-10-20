import os
from tools.ml import tf_vae
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
import configparser
import logging
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from tf_dataset.s3_image_dataset import ImageDataset
from tf_dataset.minst_dataset import MnistDataset


class MLToolMixin:
    """
    Denotes settings and paths (some from the settings) for modeling of the CNN.
    Currently it is specialized towards hyper-parameter searches to find the best configurations of an autoencoder.
    Hope to generalize more and break the hyper parameter and autoencoder stuff into their own mixins.
    """

    def __init__(self, settings_file, working_location):
        """
        Establishes the attributes needed for the run. Also checks that the required paths exist to store the models
        and tensorboard logs

        :param settings_file: path of the settings that contains paths to data and settings to determine search of
            hyper params.
        :param working_location: Consider this the root directory of the project. Paths to things like tensorboard logs
            are taken for this point. Everything you need regarding the output of your run can be found in this
            location.
        """
        if not os.path.exists(settings_file):
            raise ValueError(f"Can't find specified settings file {settings_file}")
        self.config = configparser.ConfigParser()
        self.config.read(settings_file)

        # Make sure the required subdirectories are present
        logging.info(f"Working Location: {working_location}")
        assert os.path.exists(os.path.join(working_location, 'model_checkpoints'))
        assert os.path.exists(os.path.join(working_location, 'models'))
        assert os.path.exists(os.path.join(working_location, 'settings'))
        assert os.path.exists(os.path.join(working_location, 'tensorboard_raw'))
        assert os.path.exists(os.path.join(working_location, 'study_data'))

        self.timestamp = self.config['Settings']['timestamp']
        self.Lx = int(self.config['Settings']['Lx'])
        self.Ly = int(self.config['Settings']['Ly'])
        self.feature_map_start = int(self.config['Settings']['FEATURE_MAP_START'])
        self.epochs = int(self.config['Settings']['EPOCHS'])
        batch_sizes = self.parse_int_list_from_config(self.config['Settings']['BATCH_SIZES'])
        self.hp_batch_size = hp.HParam('batch_size', hp.Discrete(batch_sizes))
        n_layers = self.parse_int_list_from_config(self.config['Settings']['N_LAYERS'])
        self.hp_n_layers = hp.HParam('n_layers', hp.Discrete(n_layers))
        feature_map_steps = self.parse_int_list_from_config(self.config['Settings']['FEATURE_MAP_STEPS'])
        self.hp_feature_map_step = hp.HParam('feature_map_step', hp.Discrete(feature_map_steps))
        stride_sizes = self.parse_int_list_from_config(self.config['Settings']['STRIDE_SIZES'])
        self.hp_stride_size = hp.HParam('stride', hp.Discrete(stride_sizes))
        #self.hp_use_batch_normalization = hp.HParam('use_batch_normalization', hp.Discrete([0, 1]))
        #self.hp_use_dropout = hp.HParam('use_dropout', hp.Discrete([0, 1]))
        self.hp_use_batch_normalization = hp.HParam('use_batch_normalization', hp.Discrete([0]))
        self.hp_use_dropout = hp.HParam('use_dropout', hp.Discrete([0]))
        #self.hp_use_dense = hp.HParam('use_dropout', hp.Discrete([1, 0]))
        self.is_image = eval(self.config['Settings']['IS_IMAGE'].title())
        self.z_dim = int(self.config['Settings']['LATENT_DIMENSION'])
        # quick run of single param or full param sweep. Use True for testing.
        self.quick_run = False
        if 'true' in self.config['Settings']['QUICK_RUN'].lower():
            self.quick_run = True
        self.variational = False
        if 'true' in self.config['Settings']['VARIATIONAL'].lower():
            self.variational = True
        self.tensorboard_debugging = False
        if 'true' in self.config['Settings']['TENSORBOARD_DEBUGGING'].lower():
            self.tensorboard_debugging = True
        self.verbose = self.config['Settings']['VERBOSE']
        self.tensorboard_sub_dir = self.config['Settings']['TENSORBOARD_SUB_DIR']
        self.checkpoint_file = os.path.join(working_location, 'model_checkpoints',
                                            'checkpoint_{}.tf'.format(self.config['Settings']['timestamp']))
        # We are saving the model as json so we have the proper structure to load the weights into. Weights come from
        # self.checkpoint_file
        self.checkpoint_json_file = os.path.join(
            working_location, 'model_checkpoints',
            'pickled_compiled_model_{}.pkl'.format(self.config['Settings']['timestamp'])
        )
        #self.debugging_logs_location = os.path.join(working_location, 'debugging_logs')
        self.best_model_file = os.path.join(
            working_location, 'models',
            'best_hyper_param_autoencoder_{}'.format(self.config['Settings']['timestamp'])
        )
        self.best_activations_file = os.path.join(
            working_location,
            'models',
            'best_hyper_param_activations_{}'.format(self.config['Settings']['timestamp'])
        )
        self.study_data_location = os.path.join(
            working_location,
            'study_data',
            self.config['Settings']['timestamp']
        )
        self.training_data_location = os.path.join(self.study_data_location, 'training_data.npy')
        self.testing_data_location = os.path.join(self.study_data_location, 'testing_data.npy')
        self.data_label_location = os.path.join(self.study_data_location, 'data_labels.npy')
        if not os.path.exists(self.study_data_location):
            os.mkdir(self.study_data_location)

        if 'generator-' in self.config['Data']['Data1']:
            self.data_train, self.data_train_labels = globals()[self.config['Data']['Data1'].split('-')[1]](train=True, train_percent=80)
            self.data_test, self.data_test_labels = globals()[self.config['Data']['Data1'].split('-')[1]](train=False, train_percent=80)
        else:
            # This will be a list of the different sources, e.g. path to transformed Z3 data, and path to transformed Z2
            # data.
            self.data = self.get_all_data_sources(self.config)
        self.checkpointer = ModelCheckpoint(
            filepath=self.checkpoint_file,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            # Want to use False but seems to be broken: https://github.com/tensorflow/tensorflow/issues/39679
            save_weights_only=False
        )
        self.run_location = working_location

        if self.tensorboard_debugging:
            tf.debugging.experimental.enable_dump_debug_info(os.path.join(working_location, 'tensorboard_raw/hp_autoencoder/'), tensor_debug_mode="FULL_HEALTH",
                                                             circular_buffer_size=-1)

        if self.quick_run:
            self.hp_batch_size = hp.HParam('batch_size', hp.Discrete([5]))
            self.hp_n_layers = hp.HParam('n_layers', hp.Discrete([3]))
            self.hp_feature_map_step = hp.HParam('feature_map_step', hp.Discrete([16]))
            self.hp_stride_size = hp.HParam('stride', hp.Discrete([1]))
            self.tensorboard_sub_dir = 'quick_run'

    @staticmethod
    def parse_int_list_from_config(string_in):
        """
        A list of integers may be provided in the settings file. Create a python list of ``int``s from the supplied
        numbers.

        :param string_in: ``str`` containing the list of numbers.
        :return: a ``list`` of ``int``s
        """
        is_split = string_in.strip(',').split(',')
        return [int(i.strip()) for i in is_split]

    def get_best_autoencoder(self):
        """
        Use this method to get the best performing model (CNN autoencoder)
        This could be a property but I'm liking the "getter" paradigm right now.

        :return: the keras model of the "best" model
        """
        if self.variational:
            return tf.keras.models.load_model(self.checkpoint_file, custom_objects={'compute_loss': tf_vae.compute_loss})
        else:
            return keras.models.load_model(self.best_model_file, custom_objects={'r_loss': r_loss})

    def get_checkpoint_model(self):
        """
        Use to get the current checkpoint model. Probably for use when a parameter sweep is running.

        :return: the keras checkpoint model in the ``model_checkpoints`` directory.
        """
        with open(self.checkpoint_json_file, 'rb') as f:
            j = pickle.load(f)
        model = keras.models.model_from_json(j)
        return model.load_weights(self.checkpoint_file)

    def get_best_activations(self):
        """
        Use this to get the activations for the best performing model.
        This could be a property but I'm liking the "getter" paradigm right now.
        
        Example of use:
        
            Consider a class with the ``MLToolMixin``. Within some method of this class you may want to get the
            activations for some set of inputs. AKA you may want to get the featur maps for some set of data.
            You can use this to get the feature maps like::
        
                >>> activation_model = self.get_best_activations()
                >>> activations = activation_model.predict(x_test)

            where ``x_test`` is the testing data. What you will get is ``activations`` which are an array of all the
            feature maps for eveyone of the "images" in x_test.

            Depending on the shape of the model you could get a feature map for a layer like::

                >>> for layer_activation in activations:
                ...     channel_image = layer_activation[current_feature, :, :, col * images_per_row + row]

            In this project we use the above in ``visualize.py``

        :return: the keras activations for the "best" model
        """
        if self.variational:
            return tf.keras.models.load_model(self.checkpoint_file, custom_objects={'compute_loss': tf_vae.compute_loss})
        else:
            return keras.models.load_model(self.best_activations_file, custom_objects={'r_loss': r_loss})

    def get_testing_data(self):
        if self.is_image:
            return self.data_test
        else:
            return np.load(self.testing_data_location)

    def get_testing_data_labels(self):
        if self.is_image:
            return self.data_test_labels
        else:
            raise ValueError('Not configured to have labels for this type of data')

    def get_training_data(self):
        if self.is_image:
            return self.data_train
        else:
            return np.load(self.training_data_location)

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



def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])


def vae_r_loss(y_true, y_pred):
    r_loss_factor = 100
    tmp_r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
    return r_loss_factor * tmp_r_loss


