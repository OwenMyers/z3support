import pickle
from aim import Run
import json
import tf_vae
import time
import os
from hashlib import sha1
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from abc import ABCMeta
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np
import argparse
import logging
from tools.ml.src.base import MLToolMixin, r_loss, vae_r_loss
from tools.ml.src.custom_callbacks import CustomCallbacks, step_decay_schedule
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Activation, Flatten, Dense, Reshape, Lambda
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
# Makes multi runs work but runs slow (same as removing tf.function decorator)
#tf.config.experimental_run_functions_eagerly(True)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class SearchTool(MLToolMixin):

    def __init__(self, settings_file, working_location):
        super().__init__(settings_file, working_location)
        self.early_stopping_patience = int(self.config['Settings']['EARLY_STOPPING_PATIENCE'])

    def train_test_model(self, run_dir, hyper_params, x_test, x_train, aim_run):

        optimizer = tf.keras.optimizers.Adam(1e-5)

        batch_size = hyper_params[self.hp_batch_size]

        train_dataset = (tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))

        epochs = self.epochs
        # set the dimensionality of the latent space to a plane for visualization later
        latent_dim = self.latent_dim
        num_examples_to_generate = 4

        # TODO revisit this section. May want to remove it depending on how we decide to create the plots
        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        random_vector_for_generation = tf.random.normal(
            shape=[num_examples_to_generate, latent_dim])
        model = tf_vae.CVAE(latent_dim)
        assert batch_size >= num_examples_to_generate
        for test_batch in test_dataset.take(1):
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]

        model.compile(loss=tf_vae.compute_loss)
        elbo = None
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            for train_x in train_dataset:
                model.train_step(model, train_x, optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(tf_vae.compute_loss(model, test_x))
            elbo = -loss.result()
            # display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch, elbo, end_time - start_time))

            aim_run.track(float(elbo.numpy()), name='loss', epoch=epoch, context={"subset": "train" })
            # generate_and_save_images(model, epoch, test_sample)

        model.predict(x_train[:1000])
        return model, float(elbo.numpy())

    def run(self, run_dir, hyper_params, x_test, x_train, aim_run):
        return self.train_test_model(run_dir, hyper_params, x_test, x_train, aim_run)

    def main(self):
        if hasattr(self, "data_train"):
            x_train = self.data_train
            x_test = self.data_test
        else:
            # DATA will contain a list of the paths to different binary data files. There should be one for each of the
            # "different types of systems" (e.g. z3, z2, high temp, etc). There is no guarantee that there are the same
            # number of configurations in each file though but the function below takes care of all of that for us to
            # make sure we get a balanced dataset and that it meets some basic requirements.
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

        c = 0
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
                                simp_hyper_params = {
                                    'hp_batch_size': batch_size,
                                    'hp_n_layers': n_layers,
                                    'hp_feature_map_step': f_map_step,
                                    'hp_stride_size': stride,
                                    'hp_use_batch_normalization': use_batch_normalization,
                                    'hp_use_dropout': use_dropout
                                }
                                c += 1
                                run_name = f"run-{c}"
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hyper_params[h] for h in hyper_params})
                                aim_run = Run()
                                run_result, loss = self.run(
                                    os.path.join(self.run_location, 'tensorboard_raw', self.tensorboard_sub_dir,
                                                 run_name),
                                    hyper_params,
                                    x_test,
                                    x_train,
                                    aim_run
                                )
                                # After each run lets attempt to log a sample of activations for the different layers
                                simp_hyper_params['loss'] = loss

                                aim_run["hparams"] = simp_hyper_params
                                if not self.tensorboard_debugging:
                                    hash_name = aim_run.hashname
                                    # Creates two output lines telling us the "asset" was created. Just a note so I
                                    # don't go digging into why later
                                    run_result.save(os.path.join(self.run_location, 'models', f'{hash_name}.tf'),
                                                    save_format='tf', save_traces=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    args = parser.parse_args()

    tool = SearchTool(args.settings, args.run_location)
    tool.main()
