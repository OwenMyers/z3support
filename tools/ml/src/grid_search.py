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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class SearchTool(MLToolMixin):
    """Grid Search made EZ

    Primary purpose of this class:

    * One: Collect high-level run operations in one place
    * Two: Inherit general ML tools from the MLMixin which abstract lower level operations
    * Three: Provide functionality for the grid search, namely the organization and visualization of performance
      of different hyperparameter models.

    **One**

    Example of a run (hopefully this feels simple):

    Example::

        python grid_search.py --settings /path/to/generated/settings/file.yml --run-location ./

    The idea being that everything important is just specified in the yaml file rather than at the command line.

    **Two**

    See the ``MLToolMixin`` docs for information on some of the lower level stuff managed by this class. Note that
    one of the things ``MLToolMixin`` does is provide simple "deserialization" of the yaml file :o. In this class
    you will see attributes with names analogous to those things provided int the settings file and they come from
    the ``MLToolMixin``.

    **Three**

    See ``main()`` method doc

    **Note**

    This class/file is meant to be run as a script.
    """

    def __init__(self, settings_file, working_location):

        super().__init__(settings_file, working_location)
        self.early_stopping_patience = int(self.config['Settings']['EARLY_STOPPING_PATIENCE'])

    def train_test_model(self, run_dir, hyper_params, x_test, x_train, aim_run):
        """Looping over the epochs happens here"""

        optimizer = tf.keras.optimizers.Adam(1e-5)

        batch_size = hyper_params[self.hp_batch_size]

        train_dataset = (tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))

        epochs = self.epochs
        # set the dimensionality of the latent space to a plane for visualization later
        latent_dim = self.latent_dim
        num_examples_to_generate = 4

        # TODO revisit this section. May want to remove it depending on how we decide to create the plots (unused!)
        # (Post TODO) keeping the random vector constant for generation (prediction) so it will be easier to see the
        # improvement.
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
        """Just runs the ``train_test_model`` method"""
        return self.train_test_model(run_dir, hyper_params, x_test, x_train, aim_run)

    def main(self):
        """Looping over hyperparameters and interfacing with ``aim``, the experiment tracker, here"""
        x_train = self.data_train
        x_test = self.data_test

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
