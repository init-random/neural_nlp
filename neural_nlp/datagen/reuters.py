import os
import tarfile
import warnings

from glob import glob
from bs4 import BeautifulSoup

from neural_nlp.io import download_data


class ReutersDataGen:

    """
    Utility class to parse Reuters-21578 SGML files and iterate over the documents. This class is a generator over the
    documents. Each document is represented as a dictionary with the fields
     - body: body text
     - title: title
     - topics: list of topics
     - mode: train or test which is derived from the split value

    TODO: parse additional fields from the SGML files.

    :param str or None split: May be cgisplit, lewissplit, modapte, or None, which is no split
    """

    _data_name = 'reuters'
    _download_url= 'http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz'

    def __init__(self, split=None):
        splits = ['cgisplit', 'lewissplit', 'modapte', None]
        if split not in splits:
            warnings.warn('UNKNOWN SPLIT TYPE')
            split = 'unknown'
        self.split = split
        self.datagen = None

    def __iter__(self):
        self.datagen = self._data_gen()
        self

    def __next__(self):
        return next(self.datagen)


    def is_training_doc(self, doc):
        is_train = False
        if self.split == 'cgisplit':
            if doc['cgisplit'] == 'TRAINING-SET':
                is_train = True
        elif self.split == 'lewissplit':
            if doc['lewissplit'] == 'TRAIN':
                is_train = True
        elif self.split == 'modapte':
            if doc['lewissplit'] == 'TRAIN' and doc['topics'] == 'YES':
                is_train = True
        return is_train

    def is_test_doc(self, doc):
        is_test = False
        if self.split == 'cgisplit':
            if doc['cgisplit'] == 'PUBLISHED-TESTSET':
                is_test = True
        elif self.split == 'lewissplit':
            if doc['lewissplit'] == 'TEST':
                is_test = True
        elif self.split == 'modapte':
            if doc['lewissplit'] == 'TEST' and doc['topics'] == 'YES':
                is_test = True
        return is_test

    def parse_doc(self, doc):
        data = {}
        text = doc.find('text')
        bt = text.find('bodytext')
        if bt is None:
            bt = ''
        else:
            bt = bt.text
        data['body'] = bt
        title = text.find('title')
        if title is not None:
            title = title.text
        data['title'] = title
        topics = doc.find('topics').find_all('d')
        if topics:
            topics = [_.text for _ in topics]
        data['topics'] = topics
        return data

    def parse_file(self, file):
        # hack: <body> tags not allowed
        sgm = open(file, 'r', encoding='latin1').read().replace('BODY>', 'BODYTEXT>')
        dom = BeautifulSoup(sgm, 'lxml')
        reuters = dom.find_all('reuters')

        for doc in reuters:
            if self.is_training_doc(doc):
                _doc = self.parse_doc(doc)
                _doc['mode'] = 'train'
                yield _doc
                # continue
            if self.is_test_doc(doc):
                _doc = self.parse_doc(doc)
                _doc['mode'] = 'test'
                yield _doc


    def _data_gen(self):
        data_dir = download_data.get_data(self._download_url, self._data_name, untar=True)
        print(data_dir)
        for filename in glob(os.path.join(data_dir, '*.sgm')):
            for doc in self.parse_file(filename):
                if doc is None:
                    continue
                yield doc

