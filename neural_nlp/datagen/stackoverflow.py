import glob
import os
from contextlib import ExitStack

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from neural_nlp.io import download_data


class StackOverflow:

    """
    This dataset is from 2015NAACL VSM-NLP workshop-"Short Text Clustering via Convolutional Neural Networks"
    See http://naacl15vs.github.io/index.html and https://github.com/jacoxu/StackOverflow.

    This utility class dowloads the stackoverflow data and providers a dask dataframe to view the data.

    :param str data_path: path to data directory
    """

    _data_name = 'stackoverflow'
    _label_dir_name = 'by_label'
    _download_urls = {'labels': 'https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/label_StackOverflow.txt',
                      'titles': 'https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText/title_StackOverflow.txt'}

    def __init__(self, data_path=None):
        self.data_path = data_path
        self._cv = None

    def filenames(self):
        return {k: v[v.rfind(os.path.sep)+1:] for k, v in self._download_urls.items()}

    def data_gen(self):
        fns = self.filenames()
        paths = {k: '%s/%s' % (download_data.get_data(self._download_urls[k], self._data_name, self.data_path), fns[k]) for k, v in fns.items()}

        with open(paths['titles'], 'r') as t_fh,\
             open(paths['labels'], 'r') as l_fh:
            for l, t in zip(l_fh, t_fh):
                yield int(l.strip()), t.strip()

    def by_label_writer(self):
        fns = self.filenames()

        dest_dir = os.path.join(self.data_path, self._data_name, self._label_dir_name)
        if os.path.exists(dest_dir):
            return
        else:
            os.makedirs(dest_dir)

        paths = {k: '%s/%s' % (download_data.get_data(self._download_urls[k], self._data_name, self.data_path), fns[k])
                 for k, v in fns.items()}

        with ExitStack() as stack, \
                open(paths['titles'], 'r') as t_fh, \
                open(paths['labels'], 'r') as l_fh:
            writers = [stack.enter_context(open(os.path.join(dest_dir, 'by_label%d.txt' % i), 'w')) for i in range(1, 21)]
            for l, t in zip(l_fh, t_fh):
                _l = int(l.strip()) - 1
                writers[_l].write('%s\n' % t.strip())

    def infinite_iter(self, fn):
        while True:
            with open(fn, 'r') as f:
                for l in f:
                    yield l

    def sample_gen(self):
        label_dir = os.path.join(self.data_path, self._data_name, self._label_dir_name, '*')

        readers = [self.infinite_iter(fn) for fn in glob.glob(label_dir)]
        for l, t in self.data_gen():
            _l = l - 1
            # get pos/neg dssm samples
            pos = next(readers[_l])
            if pos == t:
                pos = next(readers[_l])
            inp = [t, pos]
            ll = list(range(1, 21))
            ll.remove(_l + 1)
            negs = [next(readers[i - 1]) for i in np.random.choice(ll, 3)]
            inp.extend(negs)
            yield inp

    def initialize(self):
        fns = self.filenames()
        paths = ['%s' % download_data.get_data(self._download_urls[k], self._data_name, self.data_path) for k, v in fns.items()]
        self.data_path = paths[0]
        self.by_label_writer()
        self._cv = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3), stop_words='english')
        self._cv.fit((t for l, t in self.data_gen()))

    def batcher(self, batch_sz=10):
        if self._cv is None:
            raise UnboundLocalError('You must properly initialize this class running class_instance.initialize() first.')
        batch = []
        for g in self.sample_gen():
            batch.append(g)
            if len(batch) == batch_sz:
                vals = zip(*batch)
                yield [self._cv.transform(v).toarray() for v in vals]
                batch = []

