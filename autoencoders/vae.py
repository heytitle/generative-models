import tensorflow as tf


class VAE(object):
    def __init__(self, x, latent_dims):
        self.x = x
        self.latent_dims = latent_dims

        self.z, self.z_mu, self.z_log_var = self.encoder(x)
        self.logits, self.likelihoods = self.decoder(self.z)

        self.loss = VAE._loss(self.x, self.z_mu, self.z_log_var, self.logits)

    def encoder(self, x, latent_dims=2):
        with tf.variable_scope('vae-encoder'):
            h1 = tf.layers.dense(x, 512, activation=tf.nn.relu, name='l1', reuse=tf.AUTO_REUSE)
            h2 = tf.layers.dense(h1, 256, activation=tf.nn.relu, name='l2', reuse=tf.AUTO_REUSE)

            z_mu = tf.layers.dense(h2, latent_dims, activation=None, name='l-mu', reuse=tf.AUTO_REUSE)
            z_log_var = tf.layers.dense(h2, latent_dims, activation=tf.nn.softplus, name='l-log-var',
                                        reuse=tf.AUTO_REUSE)

        return VAE._sample(z_mu, z_log_var), z_mu, z_log_var

    def decoder(self, z):
        with tf.variable_scope('vae-decoder'):
            z1 = tf.layers.dense(z, 256, activation=tf.nn.relu, name='l1', reuse=tf.AUTO_REUSE)
            z2 = tf.layers.dense(z1, 512, activation=tf.nn.relu, name='l2', reuse=tf.AUTO_REUSE)

            logits = tf.layers.dense(z2, 784, activation=None, name='logit', reuse=tf.AUTO_REUSE)
            likelihoods = tf.sigmoid(logits)

        return logits, likelihoods

    @staticmethod
    def _sample(mu, log_var):
        eps = tf.random_normal(tf.shape(mu))
        return mu + tf.exp(log_var * 0.5) * eps

    @staticmethod
    def _loss(x, z_mu, z_log_var, logits):
        # Reconstruction loss (how good reconstruction is)
        recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), 1)

        # KL Divergence between latent distribution and prior distribution (normal distribution)
        kl_divergence = 0.5 * tf.reduce_sum(tf.exp(z_log_var) + tf.square(z_mu) - 1 - z_log_var, axis=1)

        return tf.reduce_mean(recon_loss + kl_divergence)



