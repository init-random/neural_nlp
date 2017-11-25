import numpy as np
import itertools
from pydash import py_


########################################################################################################################
# data generation
########################################################################################################################


class DataGen():


    def __init__(self, batch_size=50, embedding_size=128, pad_with='PAD'):

        # TODO: needs to be generalized for other data sets
        # hardcoded metadata specific to movie sentiment data
        self.sentence_length = 59
        n_tst = 1062
        self.pos_len = 5331
        self.x_len = 10662

        self.tst_idx = np.random.choice(self.x_len, n_tst, replace=False)
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.pad_with = pad_with


    def tokenizer_split(self, sentences):
        for s in sentences: yield s.strip().split()


    def get_data(self):
        import os
        print(os.getcwd())
        with open('./data/rt-polarity.pos', 'r') as fp:
            with open('./data/rt-polarity.neg', 'r') as fn:
                lines = itertools.chain(fp.readlines(), fn.readlines())
                xs = list(self.tokenizer_split(lines))
        ys = list(itertools.chain([1]*self.pos_len, [0]*self.pos_len))
        return xs, ys


    def train_test_split(self, xs, ys, vocab_map):
        pad_val = vocab_map[self.pad_with]
        _xs = py_(xs).map_(lambda s: py_(s).map_(lambda w: vocab_map[w]).value())\
                     .map(lambda s: self.pad_right(s, self.sentence_length, pad_val)).value()

        x_trn = np.array([_xs[idx] for idx in range(self.x_len) if idx not in self.tst_idx])
        x_trn = x_trn.reshape((x_trn.shape[0], self.sentence_length)).astype('int32')
        y_trn = np.array([ys[idx] for idx in range(self.x_len) if idx not in self.tst_idx]).astype('int32')
        x_tst = np.array([_xs[idx] for idx in self.tst_idx])
        x_tst = x_tst.reshape((x_tst.shape[0], self.sentence_length)).astype('int32')
        y_tst = np.array([ys[idx] for idx in self.tst_idx]).astype('int32')
        return x_trn, y_trn, x_tst, y_tst


    def get_embedding_weights(self, vocab_size):
        return np.random.uniform(-0.25, 0.25, (vocab_size, self.embedding_size)).astype('float32')


    def get_vocab_map(self, sentences, padding='PAD'):
        vocab = set([padding])
        for s in sentences:
            vocab = vocab.union(s)
        return {w: idx for idx, w in enumerate(vocab)}


    def pad_right(self, s, length, pad_with):
        pad_len = length - len(s)
        padding = [pad_with] * pad_len
        arr = itertools.chain(s, padding)
        return list(arr)


    def batch_gen(self, xs, ys, batch_size=50, shuffle=True):
        xlen = len(xs)
        perm = np.random.permutation(range(xlen))
        fst = 0
        lst = batch_size
        while fst < xlen:
            _x = xs[perm[fst:lst]]
            # just drop trailing batches
            _y = ys[perm[fst:lst]]
            fst += batch_size
            lst += batch_size
            if _x.shape[0] != batch_size: continue
            yield (_x, _y)

