import tensorflow as tf
import numpy as np
import time


class CVAEOrig(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self,
                 params,
                 latent_dim,
                 use_batch_norm=False,
                 use_dropout=False):
        if use_dropout:
            raise ValueError("Dropout not implemented in original model")
        if use_batch_norm:
            raise ValueError("Batch norm not implemented in original model")
        super(CVAEOrig, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def call(self, inputs):
        mean, log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = reparameterize(mean=mean, logvar=log_var)
        return decode(self, z)

    @tf.function
    def train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        self.gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))


class CVAEDenseOnly(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self,
                 params,
                 latent_dim,
                 use_batch_norm=False,
                 use_dropout=False):
        super(CVAEDenseOnly, self).__init__()
        self.gradients = None
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.input_edge_length = params.input_edge_length

        encoder_model = tf.keras.Sequential()
        encoder_input = tf.keras.layers.InputLayer(input_shape=(self.input_edge_length, self.input_edge_length, 1), name='encoder_input')
        encoder_model.add(encoder_input)

        if params.activation_function.lower() == 'sigmoid':
            encoder_model.add(tf.keras.layers.Activation('sigmoid'))
        elif (params.activation_function.lower() == 'none') or (params.activation_function.lower() == 'linear'):
            pass
        elif params.activation_function.lower() == 'leakyrelu':
            encoder_model.add(tf.keras.layers.LeakyReLU())
        encoder_model.add(tf.keras.layers.Flatten())
        if not len(params.hidden_layers) == 0:
            for cur_hidden_layer in params.hidden_layers:

                encoder_model.add(tf.keras.layers.Dense(cur_hidden_layer))

                # batchnorm
                if self.use_batch_norm:
                    encoder_model.add(tf.keras.layers.BatchNormalization())

                # activation
                if params.activation_function.lower() == 'sigmoid':
                    encoder_model.add(tf.keras.layers.Activation('sigmoid'))
                elif (params.activation_function.lower() == 'none') or (params.activation_function.lower() == 'linear'):
                    pass
                elif params.activation_function.lower() == 'leakyrelu':
                    encoder_model.add(tf.keras.layers.LeakyReLU())

                #dropout
                if self.use_dropout:
                    encoder_model.add(tf.keras.layers.Dropout(rate=0.25))

        encoder_model.add(tf.keras.layers.Dense(int(latent_dim + latent_dim)))
        #encoder_model.add(tf.keras.layers.LeakyReLU())
        self.encoder = encoder_model
        decoder_model = tf.keras.Sequential()
        decoder_model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        if not len(params.hidden_layers) == 0:
            for cur_hidden_layer in reversed(params.hidden_layers):
                decoder_model.add(tf.keras.layers.Dense(cur_hidden_layer))

                # batchnorm
                if self.use_batch_norm:
                    decoder_model.add(tf.keras.layers.BatchNormalization())

                # batchnorm
                if params.activation_function.lower() == 'sigmoid':
                    decoder_model.add(tf.keras.layers.Activation('sigmoid'))
                elif (params.activation_function.lower() == 'none') or (params.activation_function.lower() == 'linear'):
                    pass
                elif params.activation_function.lower() == 'leakyrelu':
                    decoder_model.add(tf.keras.layers.LeakyReLU())

                #dropout
                if self.use_dropout:
                    decoder_model.add(tf.keras.layers.Dropout(rate=0.25))

        decoder_model.add(tf.keras.layers.Dense(units=self.input_edge_length*self.input_edge_length*1)),
        #decoder_model.add(tf.keras.layers.LeakyReLU())
        decoder_model.add(tf.keras.layers.Reshape(target_shape=(self.input_edge_length, self.input_edge_length, 1))),
        if params.final_sigmoid:
            decoder_model.add(tf.keras.layers.Activation('sigmoid'))
        self.decoder = decoder_model

    def call(self, inputs):
        mean, log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = reparameterize(mean=mean, logvar=log_var)
        return decode(self, z)

    @tf.function
    def train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        self.gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))


class CVAECustom(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, params, latent_dim=2, use_dropout=False, use_batch_norm=False):

        super(CVAECustom, self).__init__()
        self.gradients = None
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.latent_dim = latent_dim
        input_dimension = params.input_edge_length
        encoder_strides_list = params.encoder_strides_list
        encoder_filters_list = params.encoder_filters_list
        encoder_kernal_list = params.encoder_kernal_list

        decoder_strides_list = params.decoder_strides_list
        decoder_filters_list = params.decoder_filters_list
        decoder_kernal_list = params.decoder_kernal_list

        # input_dimension = 12
        # encoder_strides_list = [2, 2]
        # encoder_filters_list = [5, 10]
        # encoder_kernal_list = [3, 3]

        # decoder_strides_list = [1, 2, 2]
        # decoder_filters_list = [10, 5, 1]
        # decoder_kernal_list = [3, 3, 3]

        if (len(encoder_strides_list) != len(encoder_filters_list)) or (len(encoder_filters_list) != len(encoder_kernal_list)):
            raise ValueError("Problem with strides filters or kernal list length mismatch in CVAE")
        encoder_model = tf.keras.Sequential()
        encoder_input = tf.keras.layers.InputLayer(input_shape=(input_dimension, input_dimension, 1), name='encoder_input')
        encoder_model.add(encoder_input)
        for i in range(len(encoder_strides_list)):
            conv_layer = tf.keras.layers.Conv2D(
                filters=encoder_filters_list[i],
                kernel_size=encoder_kernal_list[i],
                strides=(encoder_strides_list[i], encoder_strides_list[i]),
                #padding='same',
                name=f"encoder_conv_{i}",
                kernel_initializer="he_uniform",
            )
            encoder_model.add(conv_layer)
            if self.use_batch_norm:
                encoder_model.add(tf.keras.layers.BatchNormalization())

            if params.activation_function.lower() == 'sigmoid':
                encoder_model.add(tf.keras.layers.Activation('sigmoid'))
            elif (params.activation_function.lower() == 'none') or (params.activation_function.lower() == 'linear'):
                pass
            elif params.activation_function.lower() == 'leakyrelu':
                encoder_model.add(tf.keras.layers.LeakyReLU())

            if self.use_dropout:
                encoder_model.add(tf.keras.layers.Dropout(rate=0.25))

        encoder_model.add(tf.keras.layers.Flatten())
        encoder_model.add(tf.keras.layers.Dense(int(latent_dim + latent_dim), kernel_initializer="he_uniform"))

        #if params.activation_function.lower() == 'sigmoid':
        #    encoder_model.add(tf.keras.layers.Activation('sigmoid'))
        #elif (params.activation_function.lower() == 'none') or (params.activation_function.lower() == 'linear'):
        #    pass
        #elif params.activation_function.lower() == 'leakyrelu':
        #    encoder_model.add(tf.keras.layers.LeakyReLU())

        self.encoder = encoder_model

        decoder_model = tf.keras.Sequential()
        decoder_model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,))),
        final_layer_width = int(input_dimension/(2 * len(encoder_filters_list)))
        dense_num_units = final_layer_width**2 * decoder_filters_list[0]
        decoder_model.add(tf.keras.layers.Dense(units=dense_num_units, kernel_initializer="he_uniform"))
        #if params.activation_function.lower() == 'sigmoid':
        #    decoder_model.add(tf.keras.layers.Activation('sigmoid'))
        #elif (params.activation_function.lower() == 'none') or (
        #        params.activation_function.lower() == 'linear'):
        #    pass
        #elif params.activation_function.lower() == 'leakyrelu':
        #    decoder_model.add(tf.keras.layers.LeakyReLU())
        decoder_model.add(tf.keras.layers.Reshape(
            target_shape=(final_layer_width, final_layer_width, decoder_filters_list[0])
        ))
        for i in range(1, len(decoder_strides_list)):
            conv_transpose_layer = tf.keras.layers.Conv2DTranspose(
                filters=decoder_filters_list[i],
                kernel_size=decoder_kernal_list[i],
                strides=(decoder_strides_list[i], decoder_strides_list[i]),
                padding='same',
                name=f"decoder_conv_transpose_{i}",
                kernel_initializer="he_uniform",
            )
            decoder_model.add(conv_transpose_layer)
            if i < len(decoder_strides_list) - 1:
                if self.use_batch_norm:
                    decoder_model.add(tf.keras.layers.BatchNormalization())

                if params.activation_function.lower() == 'sigmoid':
                    decoder_model.add(tf.keras.layers.Activation('sigmoid'))
                elif (params.activation_function.lower() == 'none') or (
                        params.activation_function.lower() == 'linear'):
                    pass
                elif params.activation_function.lower() == 'leakyrelu':
                    decoder_model.add(tf.keras.layers.LeakyReLU())

                if self.use_dropout:
                    decoder_model.add(tf.keras.layers.Dropout(rate=0.25))
        if params.final_sigmoid:
            decoder_model.add(tf.keras.layers.Activation('sigmoid'))
        self.decoder = decoder_model

    def call(self, inputs):
        mean, log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = reparameterize(mean=mean, logvar=log_var)
        return decode(self, z)

    @tf.function
    def train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        self.gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))


class CVAE(tf.keras.Model):
    """
    Convolutional variational autoencoder from the tensorflow article, Doesn't work with current code setup.
    See CVAEOrig class above for a version of this that works with the current setup.
    """
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.gradients = None
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(1, 1), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(1, 1), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

#  @tf.function
#  def sample(self, eps=None):
#    if eps is None:
#      eps = tf.random.normal(shape=(100, self.latent_dim))
#    return self.decode(eps, apply_sigmoid=True)

    def call(self, inputs):
        mean, log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = reparameterize(mean=mean, logvar=log_var)
        return decode(self, z)

    @tf.function
    def train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        self.gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(self.gradients, model.trainable_variables))


def encode(model, x=None):
    mean, log_var = tf.split(model.encoder(x), num_or_size_splits=2, axis=1)
    return mean, log_var


def decode(model, z, apply_sigmoid=False):
    logits = model.decoder(z)
    if apply_sigmoid:
        probabilities = tf.sigmoid(logits)
        return probabilities
    return logits


def reparameterize(mean=None, logvar=None):
    eps = tf.random.normal(shape=tf.shape(mean))
    return eps * tf.exp(logvar * .5) + mean


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def gl_vae_r_loss(y_true, y_pred, r_loss_factor=10):
    r_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=[1, 2, 3])
    #print(f"---------------> GL Total R loss: {tf.keras.backend.sum(r_loss)}")
    return r_loss_factor * r_loss


def gl_vae_kl_loss(log_var, mu):
    kl_loss = -0.5 * tf.keras.backend.sum(1.0 + log_var - tf.keras.backend.square(mu) - tf.keras.backend.exp(log_var),
                                          axis=1)
    #print(f"---------------> GL Total KL loss: {tf.keras.backend.sum(kl_loss)}")
    return kl_loss * 0.0


def gl_compute_loss(model, x):
    mean, logvar = encode(model, x=x)
    z = reparameterize(mean=mean, logvar=logvar)
    x_logit = decode(model, z)
    r_loss = gl_vae_r_loss(x, x_logit)
    kl_loss = gl_vae_kl_loss(logvar, mean)
    return r_loss + kl_loss

# Lots of repeated code for the next several functions. Just being lazy and making a TODO to come back
# and clean this up.
def compute_loss(model, x):
    mean, logvar = encode(model, x=x)
    z = reparameterize(mean=mean, logvar=logvar)
    x_logit = decode(model, z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    #cross_ent = -vae_r_loss(x, x_logit)
    #logpx_z = cross_ent
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(500.0*logpx_z + logpz - logqz_x)


def compute_loss_breakout_px_z(model, x):
    mean, logvar = encode(model, x=x)
    z = reparameterize(mean=mean, logvar=logvar)
    x_logit = decode(model, z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return tf.reduce_mean(logpx_z)


def compute_loss_breakout_pz(model, x):
    mean, logvar = encode(model, x=x)
    z = reparameterize(mean=mean, logvar=logvar)
    x_logit = decode(model, z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return tf.reduce_mean(logpz)


def compute_loss_breakout_qz_x(model, x):
    mean, logvar = encode(model, x=x)
    z = reparameterize(mean=mean, logvar=logvar)
    x_logit = decode(model, z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return tf.reduce_mean(logqz_x)


# Trying to make a custom metric version of the above compute_loss
def metric_compute_loss(model):
    def custom_vae_loss(y_pred, y_true):
        mean, logvar = encode(model, x=y_true)
        z = reparameterize(model, mean=mean, logvar=logvar)
        x_logit = decode(model, z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        final_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

        return final_loss
    return custom_vae_loss


def main():
    optimizer = tf.keras.optimizers.Adam(1e-6)
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_size = 60000
    batch_size = 32
    test_size = 10000

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))

    epochs = 10
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 2
    num_examples_to_generate = 16

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)

    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
          train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
          loss(compute_loss(model, test_x))
        elbo = -loss.result()
        #display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        #generate_and_save_images(model, epoch, test_sample)

    model.save('./tmp_model_tf_vae.fg')


if __name__ == "__main__":
    main()
