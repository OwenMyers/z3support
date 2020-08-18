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


def train_test_model(hparams, x_test, x_train):

    input_obj = Input(shape=(L * 2, L * 2, 1))

    x = Conv2D(
        FEATURE_MAP_START,
        (3, 3),
        strides=hparams[HP_STRIDE_SIZE],
        activation='relu',
        padding='same',
        use_bias=True
    )(input_obj)

    fm = None
    for i in range(hparams[HP_N_LAYERS] - 1):
        fm = FEATURE_MAP_START + (i + 1) * hparams[HP_FEATURE_MAP_STEP]
        x = Conv2D(
            fm,
            (3, 3),
            strides=hparams[HP_STRIDE_SIZE],
            activation='relu',
            padding='same',
            use_bias=True
        )(x)
    max_fm = fm
    for i in range(hparams[HP_N_LAYERS] - 1):
        fm = max_fm - (i + 1) * hparams[HP_FEATURE_MAP_STEP]
        x = Conv2DTranspose(
            fm,
            (3, 3),
            strides=hparams[HP_STRIDE_SIZE],
            activation='relu',
            padding='same',
            use_bias=True
        )(x)

    decoded = Conv2DTranspose(
        1,
        (3, 3),
        strides=hparams[HP_STRIDE_SIZE],
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
        batch_size=hparams[HP_BATCH_SIZE],
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[
            TensorBoard(log_dir=log_dir),
            hp.KerasCallback(log_dir, hparams),
            CHECKPOINTER
        ]
    )
    _, binary_crossentropy = autoencoder.evaluate(x_test, x_test)
    return binary_crossentropy, autoencoder


def run(run_dir, hparams, x_test, x_train):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        loss, autoencoder = train_test_model(hparams, x_test, x_train)
        tf.summary.scalar('binary_crossentropy', loss, step=1)

    return autoencoder


def main():
    all_data = np.load(DATA)

    all_data = all_data.astype('float32') / 5.0
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
                    hparams = {
                        HP_BATCH_SIZE: batch_size,
                        HP_N_LAYERS: n_layers,
                        HP_FEATURE_MAP_STEP: f_map_step,
                        HP_STRIDE_SIZE: stride
                    }
                    c += 1
                    run_name = "run-%d" % c
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    autoencoder = run(os.path.join('tensorboard_raw', TENSORBOARD_SUB_DIR, run_name), hparams, x_test,
                                      x_train)

    autoencoder.load_weights(CHECKPOINT_FILE)
    autoencoder.save('models/best_hparam_autoencoder.h5')
    layers_to_encoded = int(len(autoencoder.layers) / 2)
    print(layers_to_encoded)
    layer_activations = [layer.output for layer in autoencoder.layers[:layers_to_encoded]]
    activation_model = models.Model(
        inputs=autoencoder.input,
        outputs=layer_activations
    )
    activation_model.save(os.path.join(RUN_LOCATION, 'models/best_hparam_autoencoder.h5'))
    # activations = activation_model.predict(x_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    parser.add_argument('--data', type=str,
                        help='Abs. path to data (should eventually go in the settings file)', required=True)
    args = parser.parse_args()

    # Make sure the required subdirectories are present
    assert os.path.exists(args.run_location + 'model_checkpoints')
    assert os.path.exists(args.run_location + 'models')
    assert os.path.exists(args.run_location + 'settings')
    assert os.path.exists(args.run_location + 'tensorboard_raw')
    assert os.path.exists(args.data)

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
    CHECKPOINTER = ModelCheckpoint(
        filepath=CHECKPOINT_FILE,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )
    DATA = args.data
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
