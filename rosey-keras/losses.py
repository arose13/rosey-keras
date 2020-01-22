import tensorflow.keras as k
import tensorflow.keras.backend as K


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
    (delta ** 2) * (np.sqrt(1 + (a / delta) ** 2) - 1)
    """
    return K.mean((delta ** 2) * (K.sqrt(1 + K.pow((y_true - y_pred) / delta, 2)) - 1))


def log_cosh_loss(y_true, y_pred, delta=1):
    """
    Log of the Hyperbolic Cosine. This is an approximation of the Pseudo Huber Loss
    """
    def _cosh(x):
        return (K.exp(x) + K.exp(-x)) / 2
    return K.mean(K.log(_cosh(y_pred - y_true)), axis=-1)
