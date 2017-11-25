import numpy as np
import lasagne



########################################################################################################################
# network
########################################################################################################################

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

        dropout = lasagne.layers.dropout(flattens, p=0.5)

        network = lasagne.layers.DenseLayer(dropout, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

        if self.debug == True:
            return l_in, embedding, reshape, convs, pools, flattens, dropout, network
        else:
            return network



