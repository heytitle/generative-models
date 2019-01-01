import tensorflow as tf


class VanillaAE(object):
    def __init__(self, x, latent_dims):
        self.x = x
        self.latent_dims = latent_dims

        self.z = self.encoder(x)
        self.logits, self.likelihoods = self.decoder(self.z)

        self.loss = VanillaAE._loss(self.x, self.logits, self.likelihoods)

    def encoder(self, x, latent_dims=2):
        with tf.variable_scope('vae-encoder'):
            h1 = tf.layers.dense(x, 512, activation=tf.nn.relu, name='l1', reuse=tf.AUTO_REUSE)
            h2 = tf.layers.dense(h1, 256, activation=tf.nn.relu, name='l2', reuse=tf.AUTO_REUSE)

            z = tf.layers.dense(h2, latent_dims, activation=None, name='l-mu', reuse=tf.AUTO_REUSE)

        return z

    def decoder(self, z):
        with tf.variable_scope('vae-decoder'):
            z1 = tf.layers.dense(z, 256, activation=tf.nn.relu, name='l1', reuse=tf.AUTO_REUSE)
            z2 = tf.layers.dense(z1, 512, activation=tf.nn.relu, name='l2', reuse=tf.AUTO_REUSE)

            logits = tf.layers.dense(z2, 784, activation=None, name='logit', reuse=tf.AUTO_REUSE)
            likelihoods = tf.sigmoid(logits)

        return logits, likelihoods

    @staticmethod
    def _loss(x, logits, likelihoods):
        # Reconstruction loss (how good reconstruction is)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), 1))

        # mse loss
        # recon_loss = tf.losses.mean_squared_error(x, likelihoods)

        return recon_loss
