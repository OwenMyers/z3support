import os
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.models import Model
from keras import models
from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np
import configparser
import argparse
import logging


def train_test_model(hyper_params, x_test, x_train):

    input_obj = Input(shape=(L * 2, L * 2, 1))

    x = Conv2D(
        FEATURE_MAP_START,
        (3, 3),
        strides=hyper_params[HP_STRIDE_SIZE],
        activation='relu',
        padding='same',
        use_bias=True
    )(input_obj)

    fm = None
    for i in range(hyper_params[HP_N_LAYERS] - 1):
        fm = FEATURE_MAP_START + (i + 1) * hyper_params[HP_FEATURE_MAP_STEP]
        x = Conv2D(
            fm,
            (3, 3),
            strides=hyper_params[HP_STRIDE_SIZE],
            activation='relu',
            padding='same',
            use_bias=True
        )(x)
    max_fm = fm
    for i in range(hyper_params[HP_N_LAYERS] - 1):
        fm = max_fm - (i + 1) * hyper_params[HP_FEATURE_MAP_STEP]
        x = Conv2DTranspose(
            fm,
            (3, 3),
            strides=hyper_params[HP_STRIDE_SIZE],
            activation='relu',
            padding='same',
            use_bias=True
        )(x)

    decoded = Conv2DTranspose(
        1,
        (3, 3),
        strides=hyper_params[HP_STRIDE_SIZE],
        activation='relu',
        padding='same',
        use_bias=True
    )(x)

    autoencoder = Model(input_obj, decoded)
    # autoencoder.summary()
    autoencoder.compile(
        optimizer='adadelta',
        loss='binary_crossentropy',
        metrics=['binary_crossentropy']
    )

    # log_dir='tensorboard_raw/hp_autoencoder/{}_{}'.format(datetime.now(), current_run_id)
    log_dir = os.path.join(RUN_LOCATION, 'tensorboard_raw', TENSORBOARD_SUB_DIR)
    autoencoder.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=hyper_params[HP_BATCH_SIZE],
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[
            TensorBoard(log_dir=log_dir),
            hp.KerasCallback(log_dir, hyper_params),
            CHECKPOINTER
        ]
    )
    _, binary_crossentropy = autoencoder.evaluate(x_test, x_test)
    return binary_crossentropy, autoencoder


def run(run_dir, hyper_params, x_test, x_train):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hyper_params)
        loss, autoencoder = train_test_model(hyper_params, x_test, x_train)
        tf.summary.scalar('binary_crossentropy', loss, step=1)

    return autoencoder


def import_data(list_data):
    """
    Get all the data from different systems into one place to be passed into the machine learning algorithm.

    (We will want to expand and test this if we have time or if we start doing more than just the z2 and z3 data)

    Arguments:
        Requires list of ``.npy`` files.

    This function will load ALL of the data into memory. If the data sets get too large this will be a breaking point.

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
    # At the end of this the balanced_dataset will be a list of the number of full data sets that have been passed in.
    # These still need to be put together as a single entity.
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


def main():
    # DATA will contain a list of the paths to different binary data files. There should be one for each of the
    # "different types of systems" (e.g. z3, z2, high temp, etc). There is no guarantee that there are the same number
    # of configurations in each file though but the function below takes care of all of that for us to make sure we
    # get a balanced dataset and that it meets some basic requirements.
    # all_data = np.load(DATA)
    all_data, indices = import_data(DATA)

    n_records = len(all_data)
    x_train = all_data[: n_records - int(n_records / 4)]
    x_test = all_data[int(n_records / 4):]
    x_train = np.reshape(x_train, (len(x_train), L * 2, L * 2, 1))
    x_test = np.reshape(x_test, (len(x_test), L * 2, L * 2, 1))

    with tf.summary.create_file_writer(os.path.join(RUN_LOCATION, 'tensorboard_raw', TENSORBOARD_SUB_DIR)).as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_N_LAYERS, HP_FEATURE_MAP_STEP, HP_STRIDE_SIZE],
            metrics=[hp.Metric('binary_crossentropy', display_name='Loss')]
        )

    c = 0
    autoencoder = None
    for batch_size in HP_BATCH_SIZE.domain.values:
        for n_layers in HP_N_LAYERS.domain.values:
            for f_map_step in HP_FEATURE_MAP_STEP.domain.values:
                for stride in HP_STRIDE_SIZE.domain.values:
                    hyper_params = {
                        HP_BATCH_SIZE: batch_size,
                        HP_N_LAYERS: n_layers,
                        HP_FEATURE_MAP_STEP: f_map_step,
                        HP_STRIDE_SIZE: stride
                    }
                    c += 1
                    run_name = "run-%d" % c
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hyper_params[h] for h in hyper_params})
                    autoencoder = run(os.path.join('tensorboard_raw', TENSORBOARD_SUB_DIR, run_name), hyper_params,
                                      x_test, x_train)

    autoencoder.load_weights(CHECKPOINT_FILE)
    autoencoder.save('models/best_hyper_param_autoencoder.h5')
    layers_to_encoded = int(len(autoencoder.layers) / 2)
    print(layers_to_encoded)
    layer_activations = [layer.output for layer in autoencoder.layers[:layers_to_encoded]]
    activation_model = models.Model(
        inputs=autoencoder.input,
        outputs=layer_activations
    )
    activation_model.save(os.path.join(RUN_LOCATION, 'models/best_hyper_param_autoencoder.h5'))
    # activations = activation_model.predict(x_test)


def get_all_data_sources(settings_file_parser):
    """
    Arguments:
        settings_file_parser (ConfigParser): The config parser instance containing ``DATA1, DATA2, ..., DATA<N>``

    Returns:
        A list of the full paths to all ``.npy`` files. Remember that each file contains the full list of configurations
        for that type (e.g. Z2, Z3, High temp, etc) transformed and ready for the neural network.
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    args = parser.parse_args()

    # Make sure the required subdirectories are present
    assert os.path.exists(args.run_location + 'model_checkpoints')
    assert os.path.exists(args.run_location + 'models')
    assert os.path.exists(args.run_location + 'settings')
    assert os.path.exists(args.run_location + 'tensorboard_raw')

    if not os.path.exists(args.settings):
        raise ValueError(f"Can't find specified settings file {args.settings}")
    config = configparser.ConfigParser()
    config.read(args.settings)

    L = int(config['Settings']['L'])
    FEATURE_MAP_START = int(config['Settings']['FEATURE_MAP_START'])
    EPOCHS = int(config['Settings']['EPOCHS'])
    # quick run of single param or full param sweep. Use True for testing.
    QUICK_RUN = config['Settings']['QUICK_RUN']
    VERBOSE = config['Settings']['VERBOSE']
    TENSORBOARD_SUB_DIR = config['Settings']['TENSORBOARD_SUB_DIR']
    CHECKPOINT_FILE = os.path.join(
        args.run_location,
        'checkpoint_{}.hdf5'.format(config['Settings']['timestamp'])
    )
    # This will be a list of the different sources, e.g. path to transformed Z3 data, and path to transformed Z2 data.
    DATA = get_all_data_sources(config)
    CHECKPOINTER = ModelCheckpoint(
        filepath=CHECKPOINT_FILE,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )
    RUN_LOCATION = args.run_location

    if not QUICK_RUN:
        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([15, 50]))
        HP_N_LAYERS = hp.HParam('n_layers', hp.Discrete([2, 3]))
        HP_FEATURE_MAP_STEP = hp.HParam('feature_map_step', hp.Discrete([2, 8, 16]))
        HP_STRIDE_SIZE = hp.HParam('stride', hp.Discrete([1, 2]))
    else:
        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([50]))
        HP_N_LAYERS = hp.HParam('n_layers', hp.Discrete([3]))
        HP_FEATURE_MAP_STEP = hp.HParam('feature_map_step', hp.Discrete([16]))
        HP_STRIDE_SIZE = hp.HParam('stride', hp.Discrete([1]))
        TENSORBOARD_SUB_DIR = 'quick_run'

    main()
