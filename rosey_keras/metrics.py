import tensorflow.keras as k
import tensorflow.keras.backend as K


def fraction_var_unexplained(y_true, y_pred):
    """
    Compute the unexplained variance
    Σ((y - y_hat)^2) / Σ((y - y_mean)^2)
    """
    return K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true - K.mean(y_true)))


def r2_score(y_true, y_pred):
    """
    Computes the Rsq for Keras models
    """
    return 1 - fraction_var_unexplained(y_true, y_pred)
