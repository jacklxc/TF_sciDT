import tensorflow as tf
import numpy as np
import keras.backend as K
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer
from keras.layers import SimpleRNN
class TensorAttention(Layer):
    '''
    Attention layer that operates on tensors
    '''
    input_ndim = 4
    def __init__(self, att_input_shape, context='word', init='glorot_uniform', activation='tanh', weights=None, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.context = context
        self.sample_size, self.td1, self.td2, self.wd = att_input_shape # (sample,c,w,d)
        self.initial_weights = weights
        super(TensorAttention, self).__init__(**kwargs)

    def build(self,input_shape):
        proj_dim = int(self.wd/2) # p
        self.rec_hid_dim = proj_dim
        self.att_proj = self.add_weight(name='att_proj',shape=(self.wd, proj_dim),
                                        initializer=self.init, trainable=True) # P, (d,p)
        if self.context == 'word':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(proj_dim,),initializer=self.init, trainable=True)
        elif self.context == 'clause':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.rec_hid_dim,),initializer=self.init, trainable=True)
        elif self.context == 'para':
            self.att_scorer = self.add_weight(name='att_scorer',shape=(self.td1, self.td2, proj_dim),
                                              initializer=self.init, trainable=True) # (c,w,p)
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(TensorAttention, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[3])

    def call(self, input):
        # input: D (sample,c,w,d)
        proj_input = self.activation(tf.tensordot(input, self.att_proj, axes=[[3],[0]])) # tanh(dot(D,P))=Dl,（sample,c,w,p）
        if self.context == 'word':
            att_scores = K.softmax(tf.tensordot(proj_input, self.att_scorer, axes=[[3],[0]]),axis=2) # (sample,c,w)
        elif self.context == 'clause':
            all_rnn_out = []
            RNN = SimpleRNN(self.rec_hid_dim, use_bias=False, recurrent_initializer = self.init, return_sequences=True)
            for c in range(self.td1):
                rnn_out = RNN(proj_input[:,c,:,:])
                all_rnn_out.append(K.expand_dims(rnn_out,axis=0))
            all_rnn_tensor = K.concatenate(all_rnn_out,axis=0)
            att_scores = K.softmax(tf.tensordot(K.permute_dimensions(all_rnn_tensor,(1,0,2,3)), 
                                                self.att_scorer, axes=[[3],[0]]), axis=2)
        elif self.context == 'para':
            att_scores = K.sum(tf.tensordot(proj_input, self.att_scorer, axes=[[3],[2]]), axis = [1, 2]) # (sample,c,w)
        return K.batch_dot(att_scores,input,axes=[2,2]) # (sample,c,d)

    def get_config(self):
        return {'cache_enabled': True,
                'custom_name': 'tensorattention',
                'input_shape': (self.td1, self.td2, self.wd),
                'context': self.context,
                'name': 'TensorAttention',
                'trainable': True}
