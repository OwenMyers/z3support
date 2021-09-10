import os
from abc import ABCMeta
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Activation, Flatten, Dense, Reshape
from keras.models import Model
from keras import models
from keras.callbacks import TensorBoard, EarlyStopping
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np
import argparse
import logging
import keras
from tools.ml.src.base import MLToolMixin
from tools.ml.src.custom_callbacks import CustomCallbacks


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
METRIC_ACCURACY = 'accuracy'
UPDATE_FREQ = 600,
METRICS = [
    hp.Metric(
        "epoch_accuracy",
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "batch_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "batch_loss",
        group="train",
        display_name="loss (train)",
    ),
]


class SearchTool(MLToolMixin):

    def __init__(self, settings_file, working_location):
        super().__init__(settings_file, working_location)
        self.early_stopping_patience = int(self.config['Settings']['EARLY_STOPPING_PATIENCE'])

    def train_test_model(self, run_dir, hyper_params, x_test, x_train):
        """
        This method constructs the model that will be trained and then trains it. We don't know the best model
        architecture apriori so we will be searching across different architectures.

        Arguments:
            hyper_params (Hparam): used to set the current hyper parameter that we are looping over. E.g.
            ``hyper_params[self.hp_stride_size] will get the stride we currently want to build the model with
            determined by where we are in the hyper parameter loop.
            x_test: testing dataset
            x_train: training dataset
        """

        if self.is_image:
            input_obj = Input(shape=(self.Lx, self.Ly, 1))
        else:
            input_obj = Input(shape=(self.Lx * 2, self.Ly * 2, 1))

        x = Conv2D(
            self.feature_map_start,
            (3, 3),
            strides=1,
            padding='same',
        )(input_obj)
        if hyper_params[self.hp_use_batch_normalization]:
            x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        fm = None
        for i in range(hyper_params[self.hp_n_layers]):
            if i == hyper_params[self.hp_n_layers] - 1:
                print("Done expanding feature map for now")
            else:
                fm = self.feature_map_start + (i + 1) * hyper_params[self.hp_feature_map_step]
            x = Conv2D(
                fm,
                (3, 3),
                strides=hyper_params[self.hp_stride_size],
                padding='same',
            )(x)
            x = LeakyReLU()(x)
            if hyper_params[self.hp_use_batch_normalization]:
                x = BatchNormalization()(x)
            if hyper_params[self.hp_use_dropout]:
                x = Dropout(rate=0.25)(x)
        max_fm = fm
        #if hyper_params[self.use_dense]:
        shape_before_flattening = keras.backend.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(2, name='dense_encoder_output')(x)
        x = Dense(np.prod(shape_before_flattening))(x)
        x = Reshape(shape_before_flattening)(x)

        for i in range(hyper_params[self.hp_n_layers]):
            if i == 0:
                fm = max_fm
                x = Conv2DTranspose(
                    fm,
                    (3, 3),
                    strides=1,
                    activation='relu',
                    padding='same',
                    use_bias=True
                )(x)
                x = LeakyReLU()(x)
                if hyper_params[self.hp_use_batch_normalization]:
                    x = BatchNormalization()(x)
                if hyper_params[self.hp_use_dropout]:
                    x = Dropout(rate=0.25)(x)
            else:
                fm = max_fm - i * hyper_params[self.hp_feature_map_step]
            x = Conv2DTranspose(
                fm,
                (3, 3),
                strides=hyper_params[self.hp_stride_size],
                activation='relu',
                padding='same',
                use_bias=True
            )(x)
            x = LeakyReLU()(x)
            if hyper_params[self.hp_use_batch_normalization]:
                x = BatchNormalization()(x)
            if hyper_params[self.hp_use_dropout]:
                x = Dropout(rate=0.25)(x)

        decoded = Conv2DTranspose(
            1,
            (3, 3),
            strides=1,
            activation='relu',
            padding='same',
            use_bias=True
        )(x)
        x = Activation('sigmoid')(x)

        autoencoder = Model(input_obj, decoded)
        # autoencoder.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        autoencoder.compile(
            #optimizer='adadelta',
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        autoencoder.summary()
        kwargs = {}
        if self.is_image:
            # A possible workaround for breakage in shapes (but seems to currently be working) : https: // github.com / tensorflow / tensorflow / issues / 32912
            x_train = x_train.prefetch(tf.data.AUTOTUNE)
            x_train = x_train.batch(hyper_params[self.hp_batch_size], drop_remainder=True)

            kwargs.update({
                'generator': x_train,
                #'validation_data': (x_test, x_test)
            })
        else:
            kwargs.update({
                'x': x_train,
                'y': x_train,
                'validation_data': (x_test, x_test),
                'batch_size': hyper_params[self.hp_batch_size],
            })
        kwargs.update({
            'epochs':self.epochs,
            'shuffle':True,
            'callbacks':[
                TensorBoard(
                    run_dir,
                    update_freq=UPDATE_FREQ,
                    profile_batch=0,
                    histogram_freq=2,
                    embeddings_freq=2,
                ),
                hp.KerasCallback(run_dir, hyper_params),
                self.checkpointer,
                EarlyStopping(monitor='loss', patience=self.early_stopping_patience),
                CustomCallbacks(autoencoder.to_json(), self.checkpoint_json_file)
            ]
        })

        if self.is_image:
            result = autoencoder.fit_generator(**kwargs)
        else:
            result = autoencoder.fit(**kwargs)

    def run(self, run_dir, hyper_params, x_test, x_train):
        self.train_test_model(run_dir, hyper_params, x_test, x_train)

    def main(self):
        if hasattr(self, "data_train"):
        #if isinstance(self.data, ABCMeta):
            x_train = self.data_train
            x_test = self.data_test
        else:
            # DATA will contain a list of the paths to different binary data files. There should be one for each of the
            # "different types of systems" (e.g. z3, z2, high temp, etc). There is no guarantee that there are the same
            # number of configurations in each file though but the function below takes care of all of that for us to make
            # sure we get a balanced dataset and that it meets some basic requirements.
            # all_data = np.load(DATA)
            all_data, data_labels = self.import_data(self.data)

            n_records = len(all_data)
            x_train = all_data[: n_records - int(n_records / 4)]
            x_test = all_data[int(n_records / 4):]
            x_train = np.reshape(x_train, (len(x_train), self.L * 2, self.L * 2, 1))
            x_test = np.reshape(x_test, (len(x_test), self.L * 2, self.L * 2, 1))

            np.save(self.training_data_location, x_train)
            np.save(self.testing_data_location, x_test)
            np.save(self.data_label_location, data_labels)

        with tf.summary.create_file_writer(
            os.path.join(self.run_location, 'tensorboard_raw', self.tensorboard_sub_dir)
        ).as_default():
            hp.hparams_config(
                hparams=[self.hp_batch_size, self.hp_n_layers, self.hp_feature_map_step, self.hp_stride_size, self.hp_use_batch_normalization, self.hp_use_dropout],
                metrics=METRICS
            )

        c = 0
        autoencoder = None
        for batch_size in self.hp_batch_size.domain.values:
            for n_layers in self.hp_n_layers.domain.values:
                for f_map_step in self.hp_feature_map_step.domain.values:
                    for stride in self.hp_stride_size.domain.values:
                        for use_batch_normalization in self.hp_use_batch_normalization.domain.values:
                            for use_dropout in self.hp_use_dropout.domain.values:
                                hyper_params = {
                                    self.hp_batch_size: batch_size,
                                    self.hp_n_layers: n_layers,
                                    self.hp_feature_map_step: f_map_step,
                                    self.hp_stride_size: stride,
                                    self.hp_use_batch_normalization: use_batch_normalization,
                                    self.hp_use_dropout: use_dropout
                                }
                                c += 1
                                run_name = f"run-{c}"
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hyper_params[h] for h in hyper_params})
                                self.run(
                                    os.path.join(self.run_location, 'tensorboard_raw', self.tensorboard_sub_dir, run_name),
                                    hyper_params,
                                    x_test,
                                    x_train
                                )
                                # After each run lets attempt to log a sample of activations for the different layers

        best_autoencoder = keras.models.load_model(self.checkpoint_file)
        best_autoencoder.save(self.best_model_file)
        # Get the encoder piece of the autoencoder. We call this the "activation model". This is the full model up to
        # the bottle neck.
        layers_to_encoded = int(len(best_autoencoder.layers) / 2 + 1)
        print(layers_to_encoded)
        layer_activations = [layer.output for layer in best_autoencoder.layers[:layers_to_encoded]]
        activation_model = models.Model(
            inputs=best_autoencoder.input,
            outputs=layer_activations
        )
        activation_model.save(self.best_activations_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    args = parser.parse_args()

    tool = SearchTool(args.settings, args.run_location)
    tool.main()
