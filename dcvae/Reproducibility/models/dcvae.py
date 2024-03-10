from src.libraries import *

class dcvae(cvae):
    def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), prefix='dcvae'):
        super(dcvae, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides,
                                    prefix=prefix)

    def dcvae_loss(self, batch, beta=1.0):
        mean_z, logvar_z, z_sample, x_pred = self.forward(batch)

        logpx_z = log_bernouli_pdf(x_pred, batch['x'])
        logpx_z = tf.reduce_sum(logpx_z, axis=[1, 2, 3])

        kl_divergence = tf.reduce_sum(kl_divergence_standard_prior(mean_z, logvar_z), axis=1)
        tc_loss = total_correlation(mean_z, logvar_z, z_sample)

        elbo = tf.reduce_mean(logpx_z - (kl_divergence + (beta - 1) * tc_loss))

        return elbo, tf.reduce_mean(logpx_z), tf.reduce_mean(kl_divergence)