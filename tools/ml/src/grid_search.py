import os
from tensorflow.keras.utils import plot_model
from gdl_code_repeate.vae_model import VariationalAutoencoder
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
disable_eager_execution()


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
METRIC_ACCURACY = 'accuracy'
UPDATE_FREQ = 600
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

#def load_mnist():
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#    x_train = x_train.astype('float32') / 255.
#    x_train = x_train.reshape(x_train.shape + (1,))
#    x_test = x_test.astype('float32') / 255.
#    x_test = x_test.reshape(x_test.shape + (1,))
#
#    return (x_train, y_train), (x_test, y_test)

class SearchTool(MLToolMixin):

    def __init__(self, settings_file, working_location):
        super().__init__(settings_file, working_location)
        self.early_stopping_patience = int(self.config['Settings']['EARLY_STOPPING_PATIENCE'])

    @staticmethod
    def variational_sampling(args):
        mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mu), mean=0.0, stddev=1.0)
        return mu + K.exp(log_var / 2.0) * epsilon

    def train_test_model(self, run_dir, hyper_params, x_test, x_train):
        vae = VariationalAutoencoder(
            input_dim=(28, 28, 1)
            , encoder_conv_filters=[32, 64, 64, 64]
            , encoder_conv_kernel_size=[3, 3, 3, 3]
            , encoder_conv_strides=[1, 2, 2, 1]
            , decoder_conv_t_filters=[64, 64, 32, 1]
            , decoder_conv_t_kernel_size=[3, 3, 3, 3]
            , decoder_conv_t_strides=[1, 2, 2, 1]
            , z_dim=2
        )
        learning_rate = 0.0005
        self.learning_rate = learning_rate
        r_loss_factor = 1000

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss = -0.5 * K.sum(1 + vae.log_var - K.square(vae.mu) - K.exp(vae.log_var), axis=1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return r_loss + kl_loss

        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        vae.model.compile(optimizer=optimizer, loss=vae_loss,  metrics=[vae_r_loss, vae_kl_loss])

        vae.model.fit(
            x_train,
            x_train,
            batch_size=32,
            shuffle=True,
            epochs=10,
            initial_epoch=0
        )
        return vae.model

    #def train_test_model(self, run_dir, hyper_params, x_test, x_train):
    #    """
    #    This method constructs the model that will be trained and then trains it. We don't know the best model
    #    architecture apriori so we will be searching across different architectures.

    #    Arguments:
    #        hyper_params (Hparam): used to set the current hyper parameter that we are looping over. E.g.
    #        ``hyper_params[self.hp_stride_size] will get the stride we currently want to build the model with
    #        determined by where we are in the hyper parameter loop.
    #        x_test: testing dataset
    #        x_train: training dataset
    #    """

    #    if self.is_image:
    #        input_obj = Input(shape=(self.Lx, self.Ly, 1))
    #    else:
    #        input_obj = Input(shape=(self.Lx * 2, self.Ly * 2, 1))


    #    x = Conv2D(
    #        self.feature_map_start,
    #        (3, 3),
    #        strides=1,
    #        padding='same',
    #    )(input_obj)
    #    if hyper_params[self.hp_use_batch_normalization]:
    #        x = BatchNormalization()(x)
    #    x = LeakyReLU()(x)

    #    fm = None
    #    for i in range(hyper_params[self.hp_n_layers]):
    #        if i == hyper_params[self.hp_n_layers] - 1:
    #            print("Done expanding feature map for now")
    #        else:
    #            fm = self.feature_map_start + (i + 1) * hyper_params[self.hp_feature_map_step]
    #        x = Conv2D(
    #            fm,
    #            (3, 3),
    #            strides=hyper_params[self.hp_stride_size],
    #            padding='same',
    #        )(x)
    #        x = LeakyReLU()(x)
    #        if hyper_params[self.hp_use_batch_normalization]:
    #            x = BatchNormalization()(x)
    #        if hyper_params[self.hp_use_dropout]:
    #            x = Dropout(rate=0.25)(x)
    #    x = Conv2D(
    #        fm,
    #        (3, 3),
    #        strides=1,
    #        padding='same',
    #    )(x)
    #    x = LeakyReLU()(x)
    #    max_fm = fm
    #    #if hyper_params[self.use_dense]:
    #    shape_before_flattening = tf.keras.backend.int_shape(x)[1:]
    #    x = Flatten()(x)
    #    if self.variational:
    #        self.mu = Dense(self.z_dim, name='mu')(x)
    #        self.log_var = Dense(self.z_dim, name='log_var')(x)
    #        encoder_mu_log_var = Model(input_obj, (self.mu, self.log_var))
    #        encoder_output = Lambda(self.variational_sampling, name='encoder_output')([self.mu, self.log_var])
    #        encoder = Model(input_obj, encoder_output)

    #        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
    #        x = Dense(np.prod(shape_before_flattening))(decoder_input)
    #    else:
    #        x = Dense(self.z_dim, name='dense_encoder_output')(x)
    #        x = Dense(np.prod(shape_before_flattening))(x)
    #    x = Reshape(shape_before_flattening)(x)

    #    for i in range(hyper_params[self.hp_n_layers]):
    #        if i == 0:
    #            fm = max_fm
    #            x = Conv2DTranspose(
    #                fm,
    #                (3, 3),
    #                strides=1,
    #                activation='relu',
    #                padding='same',
    #                use_bias=True
    #            )(x)
    #            x = LeakyReLU()(x)
    #            if hyper_params[self.hp_use_batch_normalization]:
    #                x = BatchNormalization()(x)
    #            if hyper_params[self.hp_use_dropout]:
    #                x = Dropout(rate=0.25)(x)
    #        else:
    #            fm = max_fm - i * hyper_params[self.hp_feature_map_step]
    #        x = Conv2DTranspose(
    #            fm,
    #            (3, 3),
    #            strides=hyper_params[self.hp_stride_size],
    #            activation='relu',
    #            padding='same',
    #            use_bias=True
    #        )(x)
    #        x = LeakyReLU()(x)
    #        if hyper_params[self.hp_use_batch_normalization]:
    #            x = BatchNormalization()(x)
    #        if hyper_params[self.hp_use_dropout]:
    #            x = Dropout(rate=0.25)(x)

    #    x = Conv2DTranspose(
    #        1,
    #        (3, 3),
    #        strides=1,
    #        activation='relu',
    #        padding='same',
    #        use_bias=True
    #    )(x)
    #    decoded = Activation('sigmoid')(x)
    #    if self.variational:
    #        decoder = Model(decoder_input, decoded)
    #        model_output = decoder(encoder_output)
    #        autoencoder = Model(input_obj, model_output)
    #    else:
    #        autoencoder = Model(input_obj, decoded)
    #    # autoencoder.summary()
    #    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    #    if self.variational:
    #        def vae_kl_loss(y_true, y_pred):
    #            kl_loss = -0.5 * K.sum(1.0 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
    #            return kl_loss

    #        def raw_log_var(y_true, y_pred):
    #            return self.log_var

    #        def raw_mu(y_true, y_pred):
    #            return self.mu

    #        def raw_square_mu(y_true, y_pred):
    #            return K.square(self.mu)

    #        def vae_loss(y_true, y_pred):
    #            new_r_loss = vae_r_loss(y_true, y_pred)
    #            kl_loss = vae_kl_loss(y_true, y_pred)
    #            return new_r_loss + kl_loss

    #        loss_in = vae_loss
    #    else:
    #        loss_in = r_loss

    #    autoencoder.compile(
    #        #optimizer='adadelta',
    #        optimizer=optimizer,
    #        #loss='binary_crossentropy',
    #        loss=loss_in,
    #        #metrics=['accuracy']
    #        metrics=[vae_kl_loss, vae_r_loss, raw_log_var, raw_mu, raw_square_mu]
    #    )
    #    autoencoder.summary()
    #    plot_model(autoencoder, to_file=os.path.join('figures', 'model_visual.png'), show_shapes=True, show_layer_names=True)
    #    kwargs = {}
    #    if self.is_image:
    #        # A possible workaround for breakage in shapes (but seems to currently be working) : https: // github.com / tensorflow / tensorflow / issues / 32912
    #        #x_train = x_train.prefetch(tf.data.AUTOTUNE)
    #        #x_train = x_train.batch(hyper_params[self.hp_batch_size], drop_remainder=True)

    #        kwargs.update({
    #            #'validation_data': (x_test, x_test)
    #            'x': x_train,
    #            'y': x_train,
    #            #'validation_data': (x_test, x_test),
    #            'batch_size': hyper_params[self.hp_batch_size],
    #        })
    #    else:
    #        kwargs.update({
    #            'x': x_train,
    #            'y': x_train,
    #            'validation_data': (x_test, x_test),
    #            'batch_size': hyper_params[self.hp_batch_size],
    #        })
    #    callbacks = [TensorBoard(
    #                    run_dir,
    #                    update_freq=UPDATE_FREQ,
    #                    profile_batch=0,
    #                    histogram_freq=2,
    #                    #embeddings_freq=2,
    #                    write_images=True,
    #                    #write_steps_per_second=True,
    #                )]
    #    if not self.tensorboard_debugging:
    #        callbacks += [#hp.KerasCallback(run_dir, hyper_params),
    #                      self.checkpointer,
    #                      EarlyStopping(monitor='loss', patience=self.early_stopping_patience),
    #                      CustomCallbacks(autoencoder.to_json(), self.checkpoint_json_file),
    #                      step_decay_schedule(initial_lr=0.0005, decay_factor=1.0, step_size=1)]
    #    kwargs.update({
    #        'epochs': self.epochs,
    #        'shuffle': True,
    #        'callbacks': callbacks,

    #    })

    #    autoencoder.fit(**kwargs)
    #    return autoencoder

    def run(self, run_dir, hyper_params, x_test, x_train):
        return self.train_test_model(run_dir, hyper_params, x_test, x_train)

    def main(self):
        if hasattr(self, "data_train"):
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

        #(x_train, y_train), (x_test, y_test) = load_mnist()
        #x_train = x_train[:1000]
        #x_test = x_test[:1000]
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
                                run_result = self.run(
                                    os.path.join(self.run_location, 'tensorboard_raw', self.tensorboard_sub_dir, run_name),
                                    hyper_params,
                                    x_test,
                                    x_train
                                )
                                # After each run lets attempt to log a sample of activations for the different layers
                                if not self.tensorboard_debugging:
                                    run_result.save(os.path.join(self.run_location, 'models', run_name + 'model_completion_save'))
        if not self.tensorboard_debugging:
            if self.variational:
                def vae_kl_loss(y_true, y_pred):
                    kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
                    return kl_loss

                def vae_loss(y_true, y_pred):
                    new_r_loss = vae_r_loss(y_true, y_pred)
                    kl_loss = vae_kl_loss(y_true, y_pred)
                    return new_r_loss + kl_loss
                check_loss = vae_loss
            else:
                check_loss = r_loss
            best_autoencoder = tf.keras.models.load_model(self.checkpoint_file, custom_objects={'vae_loss': check_loss, 'vae_kl_loss': vae_kl_loss, 'vae_r_loss': vae_r_loss})
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

        x1 = next(iter(x_test))
        x2 = next(iter(x_test))
        x = np.array([x1, x2])
        y = best_autoencoder.predict(x)
        #y = np.array(y[:,:,0,:])
        plt.imshow(x[0], aspect='auto', cmap='viridis')
        plt.savefig(os.path.join('figures', 'example_in.png'))
        plt.imshow(y[0], aspect='auto', cmap='viridis')
        plt.savefig(os.path.join('figures', 'example_out.png'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Run a parameter sweep to find the best autoencoder.')
    parser.add_argument('--settings', type=str, help='Settings file location', required=True)
    parser.add_argument('--run-location', type=str, help='Path you want the run to be done at', default='./')
    args = parser.parse_args()

    tool = SearchTool(args.settings, args.run_location)
    tool.main()
