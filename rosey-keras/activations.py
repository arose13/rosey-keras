import tensorflow.keras as k
import tensorflow.keras.backend as K


def robit(x, df=1):
    """
    Applies the CDF from the Student t distribution as the activation rather than a sigmoid.
    """
    from tensorflow_probability import distributions
    return distributions.StudentT(df, 0, 1).cdf(x)


def probit(x):
    """
    Applies the CDF from the Normal distribution as the activation rather than a sigmoid
    """
    from tensorflow_probability import distributions
    return distributions.Normal(0, 1).cdf(x)
