
ss = ['this place is crazy', 'the coffee smells good', 'some bears are like honey!']


def char_hasher(sentence):
    # choose a more robust tokenizer
    words = sentence.split()

    tokens = []
    for w in words:
        s = 0
        e = 3
        w = '#%s#' % w
        while True:
            tokens.append(w[s:e])
            s += 1
            e += 1
            if e > len(w): break
    return tokens


_words = set()
for s in ss:
    for w in s.split():
        _words.add(w)

vocab_len = len(_words)
vocab_map = {w: idx for idx, w in enumerate(_words)}
del _words

v_set = set()
for s in ss:
    for token in char_hasher(s):
        v_set.add(token)

hasher_map = {t: idx for idx, t in enumerate(v_set)}
hasher_len = len(hasher_map)
del v_set

row = []
col = []
val = []

for w, r_idx in vocab_map.items():
    for t in char_hasher(w):
        row.append(r_idx)
        col.append(hasher_map[t])
        val.append(1)

from scipy.sparse import coo_matrix

# this is the embedding weight matrix
embeddings = coo_matrix((val, (row, col)), shape=(vocab_len, hasher_len)).toarray()