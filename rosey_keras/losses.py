import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K


def _diff(x):
    return x[1:] - x[:-1]


def _wasserstein_distance(a, b):
    """
    Wasserstein distance function

    :param a: data points from distribution A
    :param b: data points from distribution B
    :return:
    """
    # Compute the cdf distance
    a_sorter = tf.argsort(a)
    b_sorter = tf.argsort(b)

    # Pooled value from both a and b
    all_values = K.concatenate([a, b])
    all_values = tf.sort(all_values)

    # Compute the difference between the sorted pooled values
    deltas = _diff(all_values)

    # Get the positions of the values of a and b between the 2 distributions
    a_cdf_indices = tf.searchsorted(
        tf.gather(a, a_sorter),
        all_values[:-1],
        side='right'
    )
    b_cdf_indices = tf.searchsorted(
        tf.gather(b, b_sorter),
        all_values[:-1],
        side='right'
    )

    # Calculate CDF
    a_cdf = tf.cast(a_cdf_indices / tf.size(a), 'float')
    b_cdf = tf.cast(b_cdf_indices / tf.size(b), 'float')

    # Wasserstein distance
    return K.sum(
        tf.multiply(
            K.abs(a_cdf - b_cdf),
            deltas
        )
    )


def wasserstein_loss(a_and_b, label):
    """
    This implements the truest Wasserstein distance and therefore no need to worry about how a or b is encoded

    :param a_and_b: all data
    :param label: binary labels about something is from distribution b or not
    :return: Wasserstein metric
    """
    bool_labels = K.cast(label, 'bool')

    a = tf.boolean_mask(a_and_b, ~bool_labels)
    b = tf.boolean_mask(a_and_b, bool_labels)

    return _wasserstein_distance(a, b)


def huber_loss(y_true, y_pred, delta=1):
    """
    a = y - f(x)
    0.5 * a ** 2 if np.abs(a) <= delta else (delta * np.abs(a)) - 0.5 * delta ** 2
    """
    a = y_true - y_pred
    cost_i = K.switch(
        a <= delta,
        0.5 * K.pow(a, 2),
        (delta * K.abs(a)) - 0.5 * delta ** 2
    )
    return K.mean(cost_i, axis=-1)


def pseudo_huber_loss(y_true, y_pred, delta=1):
    """
    a = y - f(x)
    (delta^2) * (np.sqrt(1 + (a / delta)^2) - 1)
    """
    return K.mean((delta ** 2) * (K.sqrt(1 + K.pow((y_true - y_pred) / delta, 2)) - 1))


def log_cosh_loss(y_true, y_pred, delta=1):
    """
    Log of the Hyperbolic Cosine. This is an approximation of the Pseudo Huber Loss
    """
    def _cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    return K.mean(K.log(_cosh(y_pred - y_true)), axis=-1)


def get_quantile_loss(quantile):
    """
    q * residual [if residual > 0]
    (q-1) * residual [if residual < 0]

    Unlike most loss functions this one needs to be called with the desired quantile to yield the loss function
    """
    def quantile_loss(y_true, quantile_hat):
        residual = y_true - quantile_hat
        tilt = K.maximum(
            quantile*residual,
            (quantile - 1) * residual
        )
        return K.mean(tilt, axis=-1)

    return quantile_loss
