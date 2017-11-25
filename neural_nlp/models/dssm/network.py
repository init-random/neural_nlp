import numpy as np
import lasagne
import theano.tensor as T
import theano
from scipy.sparse import coo_matrix



# class DotLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
#         super(DotLayer, self).__init__(incoming, **kwargs)
#         num_inputs = self.input_shape[1]
#         self.num_units = num_units
#         self.W = self.add_param(W, (num_inputs, num_units), name='W')
#
#     def get_output_for(self, input, **kwargs):
#         return T.dot(input, self.W)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], self.num_units)




class CosineMergeLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(CosineMergeLayer, self).__init__(incomings, **kwargs)


    def get_output_for(self, inputs, **kwargs):
        a = inputs[0]
        b = inputs[1]
        a /= T.sum(a * a, axis=1, keepdims=True)
        b /= T.sum(b * b, axis=1, keepdims=True)
        return T.sum(a * b, axis=1, keepdims=True)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1) # dot prod -> 1 dim


########################################################################################################################
# network
########################################################################################################################

batch_size = 2 #50
branches = 1
char_hasher_dim = 5 #19000
hidden_units = 300
semantic_feature_units = 7

input_var = T.imatrix().astype('int32')
input_var = T.itensor3().astype('int32')

x = T.imatrix()
# (batch, branches+1, char_hasher_dim): +1 because query is also a branch
l_in = lasagne.layers.InputLayer((batch_size, branches+1, char_hasher_dim))

q_in = lasagne.layers.SliceLayer(l_in, indices=0, axis=1)
# fork here to create branches
d1_in = lasagne.layers.SliceLayer(l_in, indices=1, axis=1)


h1 = lasagne.layers.DenseLayer(q_in, num_units=hidden_units, nonlinearity=lasagne.nonlinearities.tanh)
h2 = lasagne.layers.DenseLayer(h1, num_units=hidden_units, nonlinearity=lasagne.nonlinearities.tanh)
query_feature = lasagne.layers.DenseLayer(h2, num_units=semantic_feature_units, nonlinearity=lasagne.nonlinearities.tanh)



d1_h1 = lasagne.layers.DenseLayer(d1_in, num_units=hidden_units, nonlinearity=lasagne.nonlinearities.tanh)
d1_h2 = lasagne.layers.DenseLayer(h1, num_units=hidden_units, nonlinearity=lasagne.nonlinearities.tanh)
d1_feature = lasagne.layers.DenseLayer(h2, num_units=semantic_feature_units, nonlinearity=lasagne.nonlinearities.tanh)

dm = CosineMergeLayer([query_feature, d1_feature])


output = lasagne.layers.get_output(dm, input_var)
f = theano.function([input_var], output)
x_test = np.array([[[0, 2, 4, 1, 2],[0, 3, 4, 1, 2]], [[0, 2, 3, 1, 2],[3, 3, 4, 1, 2]]]).astype('int32')
f(x_test)

output = lasagne.layers.get_output(pool, x)
f = theano.function([x], output)
x_test = np.array([[0, 2, 1]]).astype('int32')
f(x_test)


l_in = lasagne.layers.InputLayer((3, 5), input_var=input_var)
embedding = lasagne.layers.EmbeddingLayer(l_in, input_size=5, #self.vocab_size,
                                      output_size=3, W=embedding_weights)


class NetworkGen():

    def __init__(self, input_var, sentence_length, vocab_size, num_filters=128, batch_size=50, word_windows=[3, 4, 5],
                 embedding_size=128, channels=1, debug=False):
        self.input_var = input_var
        self.vocab_size = vocab_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.word_windows = word_windows
        self.embedding_size = embedding_size
        self.channels = channels
        self.debug = debug


    def get_embedding_weights(self):
        return np.random.uniform(-0.25, 0.25, (self.vocab_size, self.embedding_size)).astype('float32')


    def get_network(self, embedding_weights=None):
        if embedding_weights is None:
            embedding_weights = self.get_embedding_weights()

        l_in = lasagne.layers.InputLayer((self.batch_size, self.sentence_length), input_var=self.input_var)

        embedding = lasagne.layers.EmbeddingLayer(l_in, input_size=self.vocab_size,
                                                  output_size=self.embedding_size, W=embedding_weights)

        reshape = lasagne.layers.ReshapeLayer(embedding, ([0], self.channels, 1, # 1 -> sentence_height
                                              embedding.output_shape[1] * embedding.output_shape[2]))


        _pools = []
        convs = []
        pools = []

        for window in self.word_windows:
            _conv = lasagne.layers.Conv2DLayer(
                reshape, num_filters=self.num_filters, filter_size=(1, window * self.embedding_size),
                stride=(1, self.embedding_size),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform(), pad=(0, int(window/2) * self.embedding_size))
            if (window % 2) == 0:
                _conv = lasagne.layers.SliceLayer(_conv, indices=slice(1, None), axis=-1)

            convs.append(_conv)

            _pool = lasagne.layers.MaxPool2DLayer(_conv,
                                                  pool_size=(1, self.sentence_length),
                                                  pad=(0, 0),
                                                  ignore_border=False)
            pools.append(_pool)
            _pool = lasagne.layers.shape.FlattenLayer(_pool)

            _pools.append(_pool)


        flattens = lasagne.layers.ConcatLayer(_pools)

        hidden = lasagne.layers.DenseLayer(flattens, num_units=flattens.output_shape[1], nonlinearity=lasagne.nonlinearities.tanh)

        dropout = lasagne.layers.dropout(hidden, p=0.75)

        network = lasagne.layers.DenseLayer(dropout, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

        if self.debug == True:
            return l_in, embedding, reshape, convs, pools, flattens, dropout, network
        else:
            return network



