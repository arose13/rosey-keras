import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K


def robit(x, df=1):
    """
    Applies the CDF from the Student t distribution as the activation rather than a sigmoid.

    :param x:
    :param df: degrees of freedom for the student T distribution
    :return:
    """
    from tensorflow_probability import distributions
    return distributions.StudentT(df, 0, 1).cdf(x)


def probit(x):
    """
    Applies the CDF from the Normal distribution as the activation rather than a sigmoid
    """
    from tensorflow_probability import distributions
    return distributions.Normal(0, 1).cdf(x)


def swish(x):
    """
    Google brain team new activation.
    Their experiments show that Swish tends to work better than ReLU on
    deeper models across a number of challenging data sets

    This make ReLU derivative continuous for all values of x
    """
    return x * k.activations.sigmoid(x)


def half_huber_relu(x, d=1):
    """
    Loss proposed in...
        Adversarial Explanations for Understanding Image
        Classification Decisions and Improved Neural
        Network Robustness
    https://arxiv.org/pdf/1906.02896.pdf

    This make ReLU derivative continuous for all values of x

    :param x:
    :return:
    """
    x = tf.where(x < 0, 0, x)  # Apply flat part of ReLU
    x = tf.where(x > (1/(2*d)), x - 1/(4*d), x)  # Apply linear part of ReLU
    x = tf.where((0 <= x) & (x <= (1/(2*d))), d*x**2, x)  # Apply huber part of the transformation
    return x


hhrelu = half_huber_relu
