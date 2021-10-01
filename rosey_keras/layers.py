import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K


#######################################################################################################################
# Layers
#######################################################################################################################
class SmartSplines(k.layers.Layer):
    """
    Learns the shape of the regression splines directly. This makes selecting the number of bases easier.
    Unlike in RBF layers where feature scaling really matters since the distance to centroids are measured
     across all dimensions, only 1 dimension j of X is measured to the centroid of that particular feature.
     While this increases the number of trainable parameters the problem is convex and easy to learn.

    Regularization is available and only penalizes the magnitude of a kernel.
    Bias is available but probably is not important in practise
    """

    def __init__(self, n_bases, use_bias=True, kernel_regularizer=None, bias_centering_loss=1.0, **kwargs):
        super().__init__(**kwargs)

        self.n_bases = n_bases
        self.output_dim = None
        self.use_bias = use_bias
        self.kernel_regularizer = k.regularizers.get(kernel_regularizer)
        self.bias_centering_loss = bias_centering_loss
        self.loc, self.tau, self.scale, self.bias = 4 * [None]

    def build(self, input_shape):
        super().build(input_shape)
        self.output_dim = input_shape[1]
        shape = (input_shape[1], self.n_bases)

        self.loc = self.add_weight(
            name='loc',
            shape=shape,
            initializer=k.initializers.he_uniform(),
            trainable=True
        )
        self.tau = self.add_weight(
            name='tau',
            shape=shape,
            initializer=k.initializers.ones(),
            trainable=True
        )
        self.scale = self.add_weight(
            name='scale',
            shape=shape,
            initializer=k.initializers.ones(),
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.bias = self.add_weight(
            name='bias',
            shape=input_shape[1],
            initializer=k.initializers.zeros(),
            trainable=self.use_bias
        )

    def call(self, inputs, **kwargs):
        """
        RBF activation function for each dimension p of x
        effectively transforming X to a new X

        φ_{jk}(x_j) = m_{jk} * exp[ -β_{jk} * ||x_j - μ_{jk}||^2 ]
        """
        # Transform inputs from (n, p) -> (n, p, n_bases)
        x = K.tile(
            K.expand_dims(inputs, axis=2),
            (1, 1, self.n_bases)
        )

        # Expand dimensions for easy broadcasting
        mu = K.expand_dims(self.loc, 0)
        beta = K.expand_dims(K.softplus(self.tau), 0)
        m = K.expand_dims(self.scale, 0)

        # Compute radial basis functions
        rbf = m * K.exp((-beta) * (x - mu) ** 2)
        agg_rbf = K.sum(rbf, axis=2)
        outputs = K.bias_add(agg_rbf, self.bias)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape  # (N, p)  since this layer is effectively just a feature transformer

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_bases': self.n_bases,
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'kernel_regularizer': k.regularizers.serialize(self.kernel_regularizer),
            'bias_centering_loss': self.bias_centering_loss
        })
        return config


class RBFLayer(k.layers.Layer):
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.centers, self.betas = 2*[None]
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        p = input_shape[1]

        self.centers = self.add_weight(
            name='centers',
            shape=(self.n_components, p),
            initializer=k.initializers.he_uniform(),
            trainable=True
        )
        self.betas = self.add_weight(
            name='betas',
            shape=(self.n_components,),
            initializer=k.initializers.constant(1.0),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        """
        RBF activation function
        φ = exp[-β * ||x-μ||^2]
        :param inputs:
        :param kwargs:
        :return:
        """
        centers = K.expand_dims(self.centers)
        h = K.transpose(centers - K.transpose(inputs))
        return K.exp(-self.betas * K.sum(h**2, axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_components

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_components': self.n_components
        })
        return config


class ConstantLayer(k.layers.Layer):
    """
    Extremely simple layer that just learns some constant value

    Example
    ```python
    import numpy as np
    from scipy import stats

    n = 10000
    x = stats.norm().rvs((n, 1000))
    y = stats.norm(100, 15).rvs(n)

    input_layer = k.layers.Input(shape=(x.shape[1],))
    output_layer = ConstantLayer(name='mean')(input_layer)  # Input literally does nothing at all

    model = k.models.Model(input_layer, output_layer)
    print(model.summary())

    model.compile(optimizer=k.optimizers.SGD(0.01), loss='mse')
    model.fit(
        x, y,
        validation_split=0.1,
        batch_size=1024, epochs=1000,
        callbacks=[
            k.callbacks.EarlyStopping(patience=10, verbose=True),
            k.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=True),
            k.callbacks.TerminateOnNaN(),
        ]
    )
    ```
    """
    def __init__(self, n_constants=1, regularizer=None, constraints=None, **kwargs):
        super().__init__(**kwargs)
        self.n_constants = n_constants
        self.regularizer = k.regularizers.get(regularizer)
        self.constraints = k.constraints.get(constraints)
        self.constant = None

    def build(self, input_shape):
        super().build(input_shape)

        self.constant = self.add_weight(
            name='constant',
            shape=(self.n_constants,),
            initializer=k.initializers.he_normal(),
            regularizer=self.regularizer,
            constraint=self.constraints,
            trainable=True
        )

    def call(self, inputs, **kwargs):
        return self.constant * K.ones_like(inputs)[:, :1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'regularizer': k.regularizers.serialize(self.regularizer),
            'constraints': k.constraints.serialize(self.constraints)
        })


#######################################################################################################################
# Regularization layers
#######################################################################################################################
class MVNRegularization(k.layers.Layer):
    """
    Penalises models for deviating from a mu=0 sd=1 no covariance between features.

    This is done by minimizing the model's mean negative log-likelihood or mean KL divergence
    """
    def __init__(self, weight=1.0, use='kl', **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.use = use
        self.mvn = None

    def build(self, input_shape):
        import tensorflow_probability as tfp
        dimensionality = input_shape[1]

        self.mvn = tfp.distributions.MultivariateNormalFullCovariance(
            loc=K.zeros(dimensionality),
            covariance_matrix=K.eye(dimensionality)
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        self.add_loss(self.weight * -K.mean(self.mvn.log_prob(inputs)))
        return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.python.framework_ops.tensor_shape.TensorShape(input_shape)
        try:
            input_shape = input_shape.with_rank(2)
        except ValueError:
            raise ValueError(f'Inputs to MVN -> {self.name} are not (N, p) matrix')
        return input_shape[-1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'm': self.weight,
            'use': self.use,
            'mvn': self.mvn
        })
