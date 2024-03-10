from src.libraries import *

def cross_entropy(p_dist, sample):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=p_dist, labels=sample)

def kl_divergence_standard_prior(mean, logvar):
    return -.5 * ((1 + logvar) - tf.exp(logvar) - tf.pow(mean, 2))

def kl_divergence(mean_p, mean_q, logvar_p, logvar_q):
    var_p = tf.exp(logvar_p)
    var_q = tf.exp(logvar_q)
    return .5 * (-1 + (logvar_q - logvar_p) + (tf.pow(mean_p - mean_q, 2) / var_q) + var_p / var_q)

def log_normal_pdf(mu, logvar, sample):
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((sample - mu) ** 2. * tf.exp(-logvar) + logvar + log2pi)

def log_bernouli_pdf(p_dist, sample):
    return -1 * cross_entropy(p_dist, sample)

def total_correlation(mu_true, logvar_true, mu_pred):

    logqz_i_j = log_normal_pdf(tf.expand_dims(mu_true, 0), tf.expand_dims(logvar_true, 0), tf.expand_dims(mu_pred, 1))
    logqz = tf.reduce_logsumexp(tf.reduce_sum(logqz_i_j, axis=2), axis=1)
    sigma_logq_k = tf.reduce_sum(tf.reduce_logsumexp(logqz_i_j, axis=1), axis=1)

    return logqz - sigma_logq_k