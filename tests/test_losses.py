import tensorflow as tf

from rosey_keras.losses import _wasserstein_distance


def test_wasserstein():
    different = _wasserstein_distance(
        tf.convert_to_tensor([3., 0., 1]),
        tf.convert_to_tensor([5., 6., 8.])
    ).numpy()
    assert different == 5

    identical = _wasserstein_distance(
        tf.convert_to_tensor([3., 0., 1]),
        tf.convert_to_tensor([0., 1., 3.])
    ).numpy()
    assert identical == 0.0
