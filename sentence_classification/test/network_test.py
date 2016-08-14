from theano import tensor as T
import lasagne
import theano
import numpy as np
from network import NetworkGen
from data_gen import DataGen


class NetworkGenTest:

    def __init__(self):
        batch_size = 3
        embedding_size = 5
        data_gen = DataGen(embedding_size=embedding_size)
        _xs, _ys = data_gen.get_data()

        vocab_map = data_gen.get_vocab_map(_xs)

        x_trn, y_trn, x_tst, y_tst = data_gen.train_test_split(_xs, _ys, vocab_map)
        x_trn, y_trn = x_trn[:3], y_trn[:3]
        x = T.imatrix().astype('int32')

        network_gen = NetworkGen(input_var=x, sentence_length=data_gen.sentence_length, vocab_size=len(vocab_map),
                                 batch_size=batch_size, embedding_size=embedding_size, debug=True)

        l_in, embedding, reshape, convs, pools, flattens, dropout, network = network_gen.get_network()

        self.batch_size = batch_size
        self.input_var = x
        self.channels = network_gen.channels
        self.embedding_size = data_gen.embedding_size
        self.word_windows = network_gen.word_windows
        self.sentence_length = data_gen.sentence_length
        self.num_filters = network_gen.num_filters
        self.sentence_height = 1
        self.l_in = l_in
        self.embedding = embedding
        self.reshape = reshape
        self.convs = convs
        self.pools = pools
        self.flattens = flattens
        self.dropout = dropout
        self.network = network
        self.x_tst = x_trn
        self.y_tst = y_trn


    def convolution(self, a, b):
        return a @ b[::-1]


    def network_prediction(self, layer, input_var, test_data):
        layer_prediction = lasagne.layers.get_output(layer, deterministic=True)
        layer_predict_function = theano.function([input_var], layer_prediction)
        return layer_predict_function(test_data)


    def test_input_layer(self):
        assert self.x_tst.shape == (self.batch_size, self.sentence_length)
        assert self.network_prediction(self.l_in, self.input_var, self.x_tst).shape == (self.batch_size, self.sentence_length)


    def test_embedding_layer(self):
        assert self.network_prediction(self.embedding, self.input_var, self.x_tst).shape == (self.batch_size, self.sentence_length, self.embedding_size)


    def test_reshape_layer(self):
        assert self.network_prediction(self.reshape, self.input_var, self.x_tst).shape == (self.batch_size, self.channels,
                                                                                           self.sentence_height, self.sentence_length * self.embedding_size)


    def test_convolutional_layers(self):
        for idx in range(len(self.convs)):
            conv = self.convs[idx]
            window = self.word_windows[idx]
            assert self.network_prediction(conv, self.input_var, self.x_tst).shape == (self.batch_size,
                                                                                       self.num_filters, self.sentence_height, self.sentence_length)

            # test calculation of convolution; note we are only using a linear activation function for now
            # test with an offset so as to not test data with padding
            print(window)
            offset = int(window/2) * self.embedding_size
            filter_size = window * self.embedding_size #+ offset
            if window % 2 == 0:
                layers = lasagne.layers.get_all_layers(conv)
                conv = layers[len(layers) - 2]
            np.testing.assert_almost_equal(lasagne.nonlinearities.rectify(self.convolution(self.network_prediction(self.reshape, self.input_var, self.x_tst)[0, 0, 0, 0:filter_size],
                                                  conv.W.eval()[0, 0, 0])), self.network_prediction(conv, self.input_var, self.x_tst)[0, 0, 0, int(window/2)])


    def test_pooling_layers(self):
        for idx in range(len(self.pools)):
            pool = self.pools[idx]
            # last dimension 1: max pooling over time extracts a max over the sentence
            assert self.network_prediction(pool, self.input_var, self.x_tst).shape == (self.batch_size,
                                                                                       self.num_filters, self.sentence_height, 1)

            # test max pooling over time calculation
            assert self.network_prediction(pool, self.input_var, self.x_tst)[0, 0, 0, 0] == np.max(self.network_prediction(self.convs[idx],
                                                                                                                      self.input_var, self.x_tst)[0, 0, 0, :])


    def test_flatten_layer(self):
       assert self.network_prediction(self.flattens, self.input_var, self.x_tst).shape == (self.batch_size, self.num_filters * 3)


    def test_output_layer(self):
       assert self.network_prediction(self.network, self.input_var, self.x_tst).shape == (3, 2)


