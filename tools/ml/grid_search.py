import os
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.models import Model
from keras import models
from keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np
import argparse
import logging
from tools.ml.base import MLToolMixin


class SearchTool(MLToolMixin):

    def __init__(self, settings_file, working_location):
        super().__init__(settings_file, working_location)

    def train_test_model(self, hyper_params, x_test, x_train):

        input_obj = Input(shape=(self.L * 2, self.L * 2, 1))

        x = Conv2D(
            self.feature_map_start,
            (3, 3),
            strides=hyper_params[self.hp_stride_size],
            activation='relu',
            padding='same',
            use_bias=True
        )(input_obj)

        fm = None
        for i in range(hyper_params[self.hp_n_layers] - 1):
            fm = self.feature_map_start + (i + 1) * hyper_params[self.hp_feature_map_step]
            x = Conv2D(
                fm,
                (3, 3),
                strides=hyper_params[self.hp_stride_size],
                activation='relu',
                padding='same',
                use_bias=True
            )(x)
        max_fm = fm
        for i in range(hyper_params[self.hp_n_layers] - 1):
            fm = max_fm - (i + 1) * hyper_params[self.hp_feature_map_step]
            x = Conv2DTranspose(
                fm,
                (3, 3),
                strides=hyper_params[self.hp_stride_size],
                activation='relu',
                padding='same',
                use_bias=True
            )(x)

        decoded = Conv2DTranspose(
            1,
            (3, 3),
            strides=hyper_params[self.hp_stride_size],
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
        log_dir = os.path.join(self.run_location, 'tensorboard_raw', self.tensorboard_sub_dir)
        autoencoder.fit(
            x_train, x_train,
            epochs=self.epochs,
            batch_size=hyper_params[self.hp_batch_size],
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=[
                TensorBoard(log_dir=log_dir),
                hp.KerasCallback(log_dir, hyper_params),
                self.checkpointer
            ]
        )
        _, binary_crossentropy = autoencoder.evaluate(x_test, x_test)
        return binary_crossentropy, autoencoder

    def run(self, run_dir, hyper_params, x_test, x_train):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hyper_params)
            loss, autoencoder = self.train_test_model(hyper_params, x_test, x_train)
            tf.summary.scalar('binary_crossentropy', loss, step=1)

        return autoencoder

    def main(self):
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
                hparams=[self.hp_batch_size, self.hp_n_layers, self.hp_feature_map_step, self.hp_stride_size],
                metrics=[hp.Metric('binary_crossentropy', display_name='Loss')]
            )

        c = 0
        autoencoder = None
        for batch_size in self.hp_batch_size.domain.values:
            for n_layers in self.hp_n_layers.domain.values:
                for f_map_step in self.hp_feature_map_step.domain.values:
                    for stride in self.hp_stride_size.domain.values:
                        hyper_params = {
                            self.hp_batch_size: batch_size,
                            self.hp_n_layers: n_layers,
                            self.hp_feature_map_step: f_map_step,
                            self.hp_stride_size: stride
                        }
                        c += 1
                        run_name = "run-%d" % c
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hyper_params[h] for h in hyper_params})
                        autoencoder = self.run(os.path.join('tensorboard_raw', self.tensorboard_sub_dir, run_name),
                                               hyper_params, x_test, x_train)

        autoencoder.load_weights(self.checkpoint_file)
        autoencoder.save(self.best_model_file)
        layers_to_encoded = int(len(autoencoder.layers) / 2)
        print(layers_to_encoded)
        layer_activations = [layer.output for layer in autoencoder.layers[:layers_to_encoded]]
        activation_model = models.Model(
            inputs=autoencoder.input,
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
