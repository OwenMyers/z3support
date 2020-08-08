import os
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import models
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import tensorflow as tf
import numpy as np
import pickle
from datetime import datetime
from contextlib import redirect_stdout
import uuid
from matplotlib import pyplot as plt

L = 4
FEATURE_MAP_START = 16
EPOCHS = 32
# quick run of single param or full param sweep. Use True for testing.
QRUN = True
VERBOSE = False
# result dir in the tensorboardraw directory
BOARD_SUB_DIR = 'hp_autoencoder'
CHECKPOINTER = ModelCheckpoint(
    filepath='hparam_sweep_model_checkpoint.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

if not QRUN:
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([15, 50]))
    HP_N_LAYERS = hp.HParam('n_layers', hp.Discrete([2, 3]))
    HP_FEATURE_MAP_STEP = hp.HParam('feature_map_step', hp.Discrete([2, 8, 16]))
    HP_STRIDE_SIZE = hp.HParam('stride', hp.Discrete([1, 2]))
else:
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([50]))
    HP_N_LAYERS = hp.HParam('n_layers', hp.Discrete([3]))
    HP_FEATURE_MAP_STEP = hp.HParam('feature_map_step', hp.Discrete([16]))
    HP_STRIDE_SIZE = hp.HParam('stride', hp.Discrete([1]))
    BOARD_SUB_DIR = 'qrun'


def train_test_model(hparams):

    input_obj = Input(shape=(L * 2, L * 2, 1))

    x = Conv2D(
        FEATURE_MAP_START,
        (3, 3),
        strides=hparams[HP_STRIDE_SIZE],
        activation='relu',
        padding='same',
        use_bias=True
    )(input_obj)

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

    # log_dir='tensorboardraw/hp_autoencoder/{}_{}'.format(datetime.now(), current_run_id)
    log_dir = os.path.join('tensorboardraw', BOARD_SUB_DIR)
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


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        loss, autoencoder = train_test_model(hparams)
        tf.summary.scalar('binary_crossentropy', loss, step=1)

    return autoencoder


def get_current_layer_display_grid_size(images_per_row, matrix_in):
    print(f'Layer shape {matrix_in.shape}')
    # Number of features in the feature map
    n_features = matrix_in.shape[-1]
    print(f'n_features {n_features}')
    # The feature map has shape (1, size, size, n_features).
    size = matrix_in.shape[1]
    print(f'size {size}')
    # Tiles the activation channels in this matrix
    n_cols = max([n_features // images_per_row, 1])
    print(f'n_cols {n_cols}')

    return size, n_cols, (size * n_cols, min(images_per_row, n_features) * size)


def fill_display_grid(to_fill, cur_image, cur_col, cur_row, cur_size):
    """Will modify data in reference ``to_fill``"""
    cur_image -= cur_image.mean() # Post-processes the feature to make it visually palatable
    cur_image /= cur_image.std()
    cur_image *= 64
    cur_image += 128
    cur_image = np.clip(cur_image, 0, 255).astype('uint8')
    print(f'cur_col * cur_size {cur_col * cur_size}')
    print(f'(cur_col + 1) * cur_size {(cur_col + 1) * cur_size}')
    print(f'cur_row * cur_size {cur_row * cur_size }')
    print(f'(cur_row + 1) * cur_size {(cur_row + 1) * cur_size}')
    print(f'cur_image.shape {cur_image.shape}')
    try:
        to_fill[
            cur_col * cur_size : (cur_col + 1) * cur_size, # Displays the grid
            cur_row * cur_size : (cur_row + 1) * cur_size
        ] = cur_image
    except ValueError:
        to_fill[
            cur_col * cur_size : (cur_col + 1) * cur_size, # Displays the grid
            cur_row * cur_size : (cur_row + 1) * cur_size
        ] = cur_image


def get_plottable_weights(raw_weights_from_get_layer):
    # short name
    w = raw_weights_from_get_layer
    print('in w.shape')
    print(w.shape)
    if len(w.shape) != 4:
        raise ValueError('Unexpected shape of weights')
    #w = w.reshape((w.shape[0], w.shape[1], w.shape[2]*w.shape[3])) 
    #print('shape after reshape')
    #print(w.shape)
    return w



def main():
    with open('pickle_data.pkl', 'rb') as f:
        all_data = pickle.load(f)

    all_data = all_data.astype('float32') / 5.0
    n_records = len(all_data)
    x_train = all_data[: n_records - int(n_records / 4)]
    x_test = all_data[int(n_records / 4):]
    x_train = np.reshape(x_train, (len(x_train), L * 2, L * 2, 1))
    x_test = np.reshape(x_test, (len(x_test), L * 2, L * 2, 1))

    with tf.summary.create_file_writer(os.path.join('tensorboard_raw', BOARD_SUB_DIR)).as_default():
        hp.hparams_config(
            hparams=[HP_BATCH_SIZE, HP_N_LAYERS, HP_FEATURE_MAP_STEP, HP_STRIDE_SIZE],
            metrics=[hp.Metric('binary_crossentropy', display_name='Loss')]
        )

    c = 0
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
                    autoencoder = run(os.path.join('tensorboard_raw', BOARD_SUB_DIR, run_name), hparams)

    autoencoder.load_weights('hparam_sweep_model_checkpoint.hdf5')
    autoencoder.save('models/best_hparam_autoencoder.h5')
    layers_to_encoded = int(len(autoencoder.layers) / 2)
    print(layers_to_encoded)
    layer_activations = [layer.output for layer in autoencoder.layers[:layers_to_encoded]]
    activation_model = models.Model(
        inputs=autoencoder.input,
        outputs=layer_activations
    )

    activations = activation_model.predict(x_test)
    images_per_row = 16
    layer_names = []
    for layer in autoencoder.layers[:layers_to_encoded]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot


    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        if 'input' in layer_name:
            continue
        size, n_cols, grid_dimensions = get_current_layer_display_grid_size(images_per_row, layer_activation)
        display_grid = np.zeros(grid_dimensions)
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                try:
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                except IndexError:
                    continue
                fill_display_grid(display_grid, channel_image, col, row, size)
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

        neuron = 0
        for layer_name in layer_names:
            print(layer_name)
            if 'conv2d' not in layer_name:
                continue

            weights = autoencoder.get_layer(name=layer_name).get_weights()[0]
            for_plot = get_plottable_weights(weights)

            size, n_cols, grid_dimensions = get_current_layer_display_grid_size(images_per_row, for_plot)
            display_grid = np.zeros(grid_dimensions)
            print(f'display_grid shape {display_grid.shape}')
            for col in range(n_cols):
                for row in range(images_per_row):
                    try:
                        channel_image = for_plot[:, :, neuron, col * images_per_row + row]
                    except IndexError:
                        continue
                    print(f'col {col}')
                    print(f'row {row}')
                    fill_display_grid(display_grid, channel_image, col, row, size)

            scale = 1.0 / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')


if __name__ == "__main__":
    main()
