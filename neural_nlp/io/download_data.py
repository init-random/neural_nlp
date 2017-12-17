"""
Altered from
https://github.com/scikit-learn/scikit-learn/blob/master/examples/applications/plot_out_of_core_classification.py
See SKLEARN_LICENSE
"""
import os
import tarfile

from sklearn.datasets import get_data_home
from sklearn.externals.six.moves import urllib


def get_data(download_url, data_name, data_path=None, untar=False):
    """
    This will download archive from the specified URL to data_home/data_name/.

    :param str download_url: full URL to archive
    :param str data_name: identifier for dataset
    :param str or None data_path: path to data directory. If None then sklearn.datasets.get_data_home
        is used
    :param bool untar: extract archive
    :rtype: str
    :return: full path to directory of the dataset
    """
    '''
    '''
    if data_path is None:
        data_path = os.path.join(get_data_home(), data_name)


    if download_url is None:
        return os.path.join(data_path, data_name)
    slash_idx= download_url.rfind('/') + 1
    archive_filename = download_url[slash_idx:]

    if not os.path.exists(data_path):
        # Download the dataset.
        print('downloading dataset (once and for all) into %s' % data_path)
        os.mkdir(data_path)

    def progress(blocknum, bs, size):
        total_sz_mb = '%.2f MB' % (size / 1e6)
        current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
        print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
              end='')

    archive_path = os.path.join(data_path, archive_filename)
    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(download_url, filename=archive_path,
                                   reporthook=progress)
        print('\r', end='')
        if untar:
            ext = archive_filename[(archive_filename.rfind('.') + 1):]
            print('untarring dataset...')
            tarfile.open(archive_path, 'r:%s' % ext).extractall(data_path)
        print('done.')

    return data_path

