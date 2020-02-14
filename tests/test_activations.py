import numpy as np
import pytest
import tensorflow as tf

from rosey_keras.activations import probit, robit, swish, half_huber_relu, hhrelu

large_positive_numbers = tf.convert_to_tensor([10e3, 20e3, 30e3], dtype='float')
large_negative_numbers = -large_positive_numbers


@pytest.mark.parametrize('activation_func', [swish, half_huber_relu, hhrelu])
def test_result_approximately_relu(activation_func):
    np.testing.assert_allclose(
        activation_func(large_negative_numbers),
        tf.nn.relu(large_negative_numbers)
    )
    np.testing.assert_allclose(
        activation_func(large_positive_numbers),
        tf.nn.relu(large_positive_numbers),
        atol=1
    )


@pytest.mark.parametrize('activation_func', [probit, robit])
def test_cdf_activation_functions(activation_func):
    np.testing.assert_allclose(
        activation_func(large_negative_numbers),
        0.0,
        atol=1e-3
    )
    np.testing.assert_allclose(
        activation_func(large_positive_numbers),
        1.0,
        atol=1e-3
    )