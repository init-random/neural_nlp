"""
Reuters
"""
import os
import tarfile
import warnings

from glob import glob
from bs4 import BeautifulSoup
from sklearn.datasets import get_data_home
from sklearn.externals.six.moves import urllib




class ReutersDataGen:
    '''Utility class to parse a SGML file and yield documents one at a time.'''

    def __init__(self, split=None, data_path=None):
        splits = ['cgisplit', 'lewissplit', 'modapte', None]
        if split not in splits:
            warnings.warn('UNKNOWN SPLIT TYPE')
            split = 'unknown'
        self.split = split
        self.data_path = data_path


    def is_training_doc(self, doc):
        '''
        :param doc:
        :return:
        '''
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
        '''
        :param doc:
        :return:
        '''
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
        '''
        :param doc:
        :return:
        '''
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
        '''
        :param file:
        :return:
        '''
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


    def data_dir(self):
        '''Iterate over documents of the Reuters dataset.

        The Reuters archive will automatically be downloaded and uncompressed if
        the `data_path` directory does not exist.

        Documents are represented as dictionaries with 'body' (str),
        'title' (str), 'topics' (list(str)) keys.

        '''

        archive_filename = 'reuters21578.tar.gz'
        download_url = '%s%s' % ('http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/',
                                 archive_filename)
        data_path = self.data_path
        if data_path is None:
            data_path = os.path.join(get_data_home(), 'reuters')
        if not os.path.exists(data_path):
            # Download the dataset.
            print('downloading dataset (once and for all) into %s' % data_path)
            os.mkdir(data_path)

            def progress(blocknum, bs, size):
                '''
                :param blocknum:
                :param bs:
                :param size:
                :return:
                '''
                total_sz_mb = '%.2f MB' % (size / 1e6)
                current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

            archive_path = os.path.join(data_path, archive_filename)
            urllib.request.urlretrieve(download_url, filename=archive_path,
                                       reporthook=progress)
            print('\r', end='')
            print('untarring Reuters dataset...')
            tarfile.open(archive_path, 'r:gz').extractall(data_path)
            print('done.')

        return data_path

    def data_gen(self):
        for filename in glob(os.path.join(self.data_dir(), '*.sgm')):
            for doc in self.parse_file(filename):
                if doc is None:
                    continue
                yield doc


