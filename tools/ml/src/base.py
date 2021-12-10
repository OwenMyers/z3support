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
# Your linter might show some datasets as unused but you will need them imported. We use the name supplied in the
# settings file to create an instance of the class but it has to be imported
from custom_datasets.s3_image_dataset import ImageDataset
from custom_datasets.minst_dataset import MnistDataset
from custom_datasets.physics_dataset import PhysicsDataset
from custom_datasets.artificial_dataset_0 import ArtificialDataset0
from custom_datasets.artificial_dataset_1 import ArtificialDataset1
from custom_datasets.artificial_dataset_2 import ArtificialDataset2
from custom_datasets.artificial_dataset_3 import ArtificialDataset3


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
        if int(self.config['Settings']['Lx']) != int(self.config['Settings']['Ly']):
            raise ValueError("Can't handle non-square lattice shapes")
        self.L = int(self.config['Settings']['Lx'])
        self.optimize_step_size = float(self.config['Settings']['OPTIMIZE_STEP_SIZE'])
        self.feature_map_start = int(self.config['Settings']['FEATURE_MAP_START'])
        self.epochs = int(self.config['Settings']['EPOCHS'])
        self.latent_dim = int(self.config['Settings']['LATENT_DIMENSION'])
        batch_sizes = self.parse_int_list_from_config(self.config['Settings']['BATCH_SIZES'])
        self.hp_batch_size = hp.HParam('batch_size', hp.Discrete(batch_sizes))
        n_layers = self.parse_int_list_from_config(self.config['Settings']['N_LAYERS'])
        self.hp_n_layers = hp.HParam('n_layers', hp.Discrete(n_layers))
        feature_map_steps = self.parse_int_list_from_config(self.config['Settings']['FEATURE_MAP_STEPS'])
        self.hp_feature_map_step = hp.HParam('feature_map_step', hp.Discrete(feature_map_steps))
        stride_sizes = self.parse_int_list_from_config(self.config['Settings']['STRIDE_SIZES'])
        self.hp_stride_size = hp.HParam('stride', hp.Discrete(stride_sizes))
        self.hp_use_batch_normalization = hp.HParam('use_batch_normalization', hp.Discrete(
            self.parse_bool_list_from_config(self.config['Settings']['USE_BATCH_NORMALIZATION'])
        ))
        self.hp_use_dropout = hp.HParam('use_dropout', hp.Discrete(
            self.parse_bool_list_from_config(self.config['Settings']['USE_DROPOUT'])
        ))
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
            self.data_train, self.data_train_labels = globals()[self.config['Data']['Data1'].split('-')[1]](
                train=True, train_percent=80)
            self.data_test, self.data_test_labels = globals()[self.config['Data']['Data1'].split('-')[1]](
                train=False, train_percent=80)
        else:
            # This will be a list of the different sources, e.g. path to transformed Z3 data, and path to transformed Z2
            # data.
            path_list = self.get_all_data_sources(self.config)
            self.data_train, self.data_train_labels = PhysicsDataset(
                train=True,
                train_percent=80,
                path_list=path_list,
                lattice_size=self.L,
                configuration_shape=self.config['Data']['SHAPE'].lower().strip()
            )
            self.data_test, self.data_test_labels = PhysicsDataset(
                train=False,
                train_percent=80,
                path_list=path_list,
                lattice_size=self.L,
                configuration_shape = self.config['Data']['SHAPE'].lower().strip()
            )

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
            tf.debugging.experimental.enable_dump_debug_info(os.path.join(working_location, 'tensorboard_raw/hp_autoencoder/'),
                                                             tensor_debug_mode="FULL_HEALTH",
                                                             circular_buffer_size=-1)

        if self.quick_run:
            self.hp_batch_size = hp.HParam('batch_size', hp.Discrete([5]))
            self.hp_n_layers = hp.HParam('n_layers', hp.Discrete([3]))
            self.hp_feature_map_step = hp.HParam('feature_map_step', hp.Discrete([16]))
            self.hp_stride_size = hp.HParam('stride', hp.Discrete([1]))
            self.tensorboard_sub_dir = 'quick_run'

        self.model_params = self.create_model_params()

    def create_model_params(self):
        i = 1
        cur_section_name = f'ModelParams{i}'
        while cur_section_name in self.config:
            model_params_conf_section = self.config[cur_section_name]

            if model_params_conf_section['MODEL_NAME']:


    @staticmethod
    def parse_bool_list_from_config(string_in):
        """
        A list of booleans may be provided in the settings file. Create a python list of ``bool``s from the supplied
        booleans.

        :param string_in: ``str`` containing the list of bools.
        :return: a ``list`` of ``int``s
        """

        is_split = string_in.strip(',').split(',')
        to_return = []
        for cur in is_split:
            if 'true' in cur.lower():
                to_return.append(True)
            elif 'false' in cur.lower():
                to_return.append(False)
            else:
                raise ValueError("Unexpected token in boolean list")
        return to_return

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


