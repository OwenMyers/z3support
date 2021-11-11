import tensorflow as tf
import numpy as np
import time


class CVAECustom(tf.keras.Model):
    """Convolutional variational autoencoder."""
    def __init__(self, latent_dim):

        super(CVAECustom, self).__init__()
        self.gradients = None
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(40, 40, 1)),
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
                tf.keras.layers.Dense(units=10*10*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(10, 10, 32)),
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
    """Convolutional variational autoencoder."""
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


def decode(model, z, apply_sigmoid=True):
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

# Lots of repeated code for the next several functions. Just being lazy and making a TODO to come back
# and clean this up.
def compute_loss(model, x):
    mean, logvar = encode(model, x=x)
    z = reparameterize(mean=mean, logvar=logvar)
    x_logit = decode(model, z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


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
