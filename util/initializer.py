import numpy as np

def uniform_random_weights(k_out, k_in, scale, dtype=np.float32):
    """
    Returns weights of shape (k_in, k_out) initialized between [-scale, scale]
    """
    return ((np.random.rand(k_in, k_out) * 2 - 1) * scale).astype(dtype)

def random_weights_tanh(k_out, k_in, dtype=np.float32):
    scale = (6. / (k_in + k_out)) ** .5
    return uniform_random_weights(k_out, k_in, scale, dtype=dtype)

def random_weights_sigmoid(k_out, k_in, dtype=np.float32):
    scale = 4. * (6. / (k_in + k_out)) ** .5
    return uniform_random_weights(k_out, k_in, scale, dtype=dtype)

def random_weights_reLu(k_out, k_in, dtype=np.float32):
    scale = (2. / (k_in + k_out)) ** .5
    return uniform_random_weights(k_out, k_in, scale, dtype=dtype)

def biases(k, dtype=np.float32):
    """
    Initialize biases as zero.
    """
    return np.zeros((k, ), dtype=dtype)
