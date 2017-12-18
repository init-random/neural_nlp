import numpy as np
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

from neural_nlp.datagen import stackoverflow


n_records = 20000
feature_dim = 128

loaded_model = load_model('integrations/models/model_dssm_20171217_2226_e4.h5')

so = stackoverflow.StackOverflow()
so.initialize()

########################################################################################################################
# dssm evaluation
########################################################################################################################

mtrx = np.zeros((n_records, feature_dim))

labels, titles = zip(*so.data_gen())
labels =  np.array(labels)
titles =  np.array(titles)

b_idx = 0
e_idx = 100
for q, *samples in so.batcher(batch_sz=100):
    embeddings = loaded_model.predict_on_batch(q)
    mtrx[b_idx:e_idx] = embeddings
    b_idx += 100
    e_idx += 100

mtrx_norm = mtrx**2
mtrx_norm = mtrx / np.sqrt(mtrx_norm.sum(axis=1, keepdims=True))

share = []
for i in range(20000):
    cos_sims = mtrx_norm[i] @ mtrx_norm.T
    args = labels[cos_sims.argsort()[-11:]]
    share.append(((args==args[10]).sum()-1)/10)
share = np.array(share)

print('DSSM P@10: ', share.mean())
# 0.77

########################################################################################################################
# tfidf evaluation
########################################################################################################################

_tfidf = TfidfVectorizer()
tfidf = _tfidf.fit_transform(titles)

share = []
for i in range(20000):
    cos_sims = (tfidf[i] @ tfidf.T).toarray()[0]
    args = labels[cos_sims.argsort()[-11:]]
    share.append(((args==args[10]).sum()-1)/10)
share = np.array(share)
share.mean()

print('TFIDF P@10: ', share.mean())
# 0.47

