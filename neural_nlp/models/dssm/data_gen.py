from scipy.sparse import coo_matrix

class LilIndexer:
    def __init__(self, sparse_mtrx, index_to_dense=True):
        self.sparse_mtrx = sparse_mtrx
        self.index_to_dense = index_to_dense
        self.shape = sparse_mtrx.shape

    def __getitem__(self, item):
        return self.sparse_mtrx[item].todense()

    def __call__(self, *args, **kwargs):
        return True



def line_text(s):
    t = s.lower().strip().split('\t')
    print(len(t))
    if len(t) == 1: return None
    if len(t) == 2:
        return t[1]
    return '%s %s' % (t[1], t[2])



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

ss = open('/home/k/data/dssm/curation_content.txt', 'r').readlines()
ss = [line_text(s) for s in ss if line_text(s) is not None]

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


# this is the embedding weight matrix
embeddings = LilIndexer(coo_matrix((val, (row, col)), shape=(vocab_len, hasher_len)).tolil().astype('int32'))



