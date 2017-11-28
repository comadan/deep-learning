def uniform_random_weights(k_out, k_in, scale):
    """
    Returns weights of shape (k_in, k_out) initialized between [-scale, scale]
    """
    return ((np.random.rand(k_in, k_out) * 2 - 1) * scale)

def random_weights_tanh(k_in, k_out):
    scale = (6. / (k_in + k_out)) ** .5
    return uniform_random_weights(k_out, k_in, scale)

def random_weights_sigmoid(k_out, k_in):
    scale = 4. * (6. / (k_in + k_out)) ** .5
    return uniform_random_weights(k_out, k_in, scale)

def random_weights_reLu(k_out, k_in):
    scale = (2. / (k_in + k_out)) ** .5
    return uniform_random_weights(k_out, k_in, scale)

def biases(k):
    """
    Initialize biases as zero.
    """
    return np.zeros((k, ))
