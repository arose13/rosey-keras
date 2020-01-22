import tensorflow.keras as k
import tensorflow.keras.backend as K


def keras_frac_var_unexplained(y_true, y_pred):
    """
    Compute the unexplained variance
    SumSq(y - y_hat) / SumSq(y - y_mean)
    """
    return K.sum(K.square(y_true - y_pred)) / K.sum(K.square(y_true - K.mean(y_true)))


def keras_r2(y_true, y_pred):
    """
    Computes the Rsq for Keras models
    """
    return 1 - keras_frac_var_unexplained(y_true, y_pred)
