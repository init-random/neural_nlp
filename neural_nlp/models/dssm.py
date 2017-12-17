from keras.layers import Dense, Input, dot, concatenate
from keras.models import Model
import numpy as np
import lasagne
import theano.tensor as T
import theano
from scipy.sparse import coo_matrix
import cytoolz.curried as z
from neural_nlp.toolz import toolz



def create_model(feature_dim, n_negative_samples=3):
    _hidden = Dense(250, activation='relu', name='h1')
    _encoding = Dense(128, activation='sigmoid', name='e1')

    def shared_encoding(input, hidden, encoding):
        h = hidden(input)
        e = encoding(h)
        return e

    q_input = Input(shape=(feature_dim, ), name='input')
    q_enc = shared_encoding(q_input, _hidden, _encoding)
    p_input = Input(shape=(feature_dim, ), name='p_input')
    p_enc = shared_encoding(p_input, _hidden, _encoding)

    n_inputs = []
    n_encs = []
    for n in range(n_negative_samples):
        i = Input(shape=(feature_dim, ), name='ninput%d' % n)
        n_inputs.append(i)
        n_encs.append(shared_encoding(i, _hidden, _encoding))

    inputs = [q_input, p_input] + n_inputs
    encodings = [q_enc, p_enc] + n_encs
    cos_sims = [dot([encodings[0], encodings[doc_idx]], axes=1, normalize=True) for doc_idx in range(1, len(encodings))]
    cos_sims = concatenate(cos_sims)

    output = Dense(n_negative_samples + 1, activation='softmax')(cos_sims)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    encoder = Model(inputs=q_input, outputs=encodings[0])
    encoder.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    model.summary()
    return model, encoder


