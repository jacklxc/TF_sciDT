# Simple modification to the implementation of TimeDistributedDense in keras

from keras.engine.topology import Layer
from keras import backend as K
from keras import activations, initializers, regularizers, constraints

class HigherOrderTimeDistributedDense(Layer):
    '''Apply the same dense layer on all inputs over two time dimensions.
    Useful when the input to the layer is a 4D tensor.

    # Input shape
        4D tensor with shape `(nb_sample, time_dimension1, time_dimension2, input_dim)`.

    # Output shape
        4D tensor with shape `(nb_sample, time_dimension1, time_dimension2, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 4

    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear', weights=None,
                 input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.initial_weights = weights

        self.input_dim = input_dim
        super(HigherOrderTimeDistributedDense, self).__init__(**kwargs)

    def build(self,input_shape):
        input_dim = input_shape[3]
        self.W = self.add_weight(name='W',shape=(input_dim, self.output_dim),initializer=self.init, trainable=True)
        self.b = self.add_weight(name='b',shape=(self.output_dim,),initializer=self.init, trainable=True)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(HigherOrderTimeDistributedDense, self).build(input_shape)
        
    def compute_output_shape(self,input_shape):
        input_shape = input_shape
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def call(self, X):
        def out_step(X_i, states):
            def in_step(x, in_states):
                output = K.dot(x, self.W) + self.b
                return output, []

            _, in_outputs, _ = K.rnn(in_step, X_i,
                                    initial_states=[],
                                    mask=None)
            return in_outputs, []
        _, outputs, _ = K.rnn(out_step, X,
                             initial_states=[],
                             mask=None)
        outputs = self.activation(outputs)
        return outputs
    
    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'input_dim': self.input_dim}
        base_config = super(HigherOrderTimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))