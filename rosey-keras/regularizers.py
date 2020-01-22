import tensorflow.keras as k
import tensorflow.keras.backend as K


class Fusion(k.regularizers.Regularizer):
    """
    Base class for all of the fusion regularizers.
    Fused Lasso -> https://web.stanford.edu/group/SOL/papers/fused-lasso-JRSSB.pdf
    Absolute Fused Lasso -> http://www.kdd.org/kdd2016/papers/files/rpp0343-yangA.pdf
    """

    def __init__(self, l1: float = 0, fusion: float = 0, absolute_fusion: float = 0):
        self.l1 = K.cast_to_floatx(l1)
        self.fuse = K.cast_to_floatx(fusion)
        self.abs_fuse = K.cast_to_floatx(absolute_fusion)

    def __call__(self, x):
        regularization = 0.

        x_rolled = self._roll_tensor(x)

        # Add components if they are given
        if self.l1:
            # \lambda ||x||
            regularization += self.l1 * K.sum(K.abs(x))
        if self.fuse:
            # \lambda \sum{ |x - x_+1| }
            regularization += self.fuse * K.sum(K.abs(x - x_rolled))
        if self.abs_fuse:
            # \lambda \sum{ ||x| - |x_+1|| }
            regularization += self.abs_fuse * K.sum(K.abs(K.abs(x) - K.abs(x_rolled)))

        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
            'fusion': float(self.fuse),
            'abs_fusion': float(self.abs_fuse)
        }

    @staticmethod
    def _roll_tensor(x):
        vector_length = K.int_shape(x)[0]
        x_tile = K.tile(x, [2, 1])
        return x_tile[vector_length - 1:-1]

    @staticmethod
    def check_alpha_and_ratio(alpha, l1_ratio):
        assert alpha >= 0, 'alpha must be >= 0'
        assert 0.0 <= l1_ratio <= 1.0, 'l1_ratio must be between [0, 1]'

    @staticmethod
    def check_l1_and_fusion(l1, fuse):
        assert l1 is not None and fuse is not None, 'Both l1 and fuse must be given'


def fused_lasso(alpha=2.0, l1_ratio=0.5, l1=None, fuse=None) -> k.regularizers.Regularizer:
    """
    Either alpha and l1_ratio must be given or l1 and fuse
    :param alpha: Regularization strength
    :param l1_ratio: Proportion of alpha to transfer to the l1 regularization term
    :param l1:
    :param fuse:
    :return:
    """
    if l1 is None and fuse is None:
        # Use the alpha and l1 ratio
        Fusion.check_alpha_and_ratio(alpha, l1_ratio)
        return Fusion(l1=alpha * l1_ratio, fusion=alpha * (1 - l1_ratio))

    elif l1 is not None or fuse is not None:
        Fusion.check_l1_and_fusion(l1, fuse)
        return Fusion(l1=l1, fusion=fuse)

    else:
        raise ValueError('`l1` and `fuse` must be given OR `alpha` and `l1_ratio`')


def absolute_fused_lasso(alpha=2.0, l1_ratio=0.5, l1=None, abs_fuse=None):
    """
    Either alpha and l1_ratio must be given or l1 and abs_fuse
    :param alpha: Regularization strength
    :param l1_ratio: Proportion of alpha to transfer to the l1 regularization term
    :param l1:
    :param abs_fuse:
    :return:
    """
    if l1 is None and abs_fuse is None:
        # Use the alpha and l1 ratio
        Fusion.check_alpha_and_ratio(alpha, l1_ratio)
        return Fusion(l1=alpha * l1_ratio, absolute_fusion=alpha * (1 - l1_ratio))

    elif l1 is not None or abs_fuse is not None:
        Fusion.check_l1_and_fusion(l1, abs_fuse)
        return Fusion(l1=l1, absolute_fusion=abs_fuse)

    else:
        raise ValueError('`l1` and `fuse` must be given OR `alpha` and `l1_ratio`')
