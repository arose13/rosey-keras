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

        \varphi_{jk}(x_j) = m_{jk} * exp[ -\beta_{jk} * ||x_j - \mu_{jk}||^2 ]
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


#######################################################################################################################
# Regularization layers
#######################################################################################################################
class MVNRegularization(k.layers.Layer):
    """
    Penalises models for deviating from a mu=0 sd=1 no covariance between features.

    This is done by minimizing the model's mean negative log-likelihood
    """

    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
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
            'mvn': self.mvn
        })
