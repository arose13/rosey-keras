import numpy as np
import tensorflow as tf

from hypothesis import given, settings
from hypothesis import strategies as some

from rosey_keras.losses import _wasserstein_distance


def test_wasserstein_known_output():
    different = _wasserstein_distance(
        tf.convert_to_tensor([3, 0, 1], dtype='float'),
        tf.convert_to_tensor([5, 6, 8], dtype='float')
    ).numpy()
    assert np.isclose(different, 5)

    identical = _wasserstein_distance(
        tf.convert_to_tensor([3, 0, 1, 5, 6, 8], dtype='float'),
        tf.convert_to_tensor([5, 6, 8, 3, 0, 1], dtype='float')
    ).numpy()
    assert np.isclose(identical, 0)


@given(
    a=some.lists(
        some.floats(min_value=-1e9, max_value=1e9),
        min_size=3, max_size=1000
    ),
    b=some.lists(
        some.floats(min_value=-1e9, max_value=1e9),
        min_size=3, max_size=1000
    )
)
def test_wasserstein_is_equivalent(a, b):
    from scipy import stats

    tf_distance = _wasserstein_distance(
        tf.convert_to_tensor(a, dtype='float'),
        tf.convert_to_tensor(b, dtype='float')
    ).numpy()

    scipy_distance = stats.wasserstein_distance(a, b)

    assert np.isclose(tf_distance, scipy_distance)
