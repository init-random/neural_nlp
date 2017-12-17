import datetime
import os


def dict_to_str(d, prefix=''):
    kvs = []
    for k, v in d.items():
        if isinstance(v, dict):
            v = '\n' + dict_to_str(v, prefix='\t')
        kv = '%s%s -> %s' % (prefix, str(k), str(v))
        kvs.append(kv)
    return '\n'.join(kvs)


class MetaData(object):

    def __init__(self, identifier, model, timestamp=None, misc=None):
        self.identifier = identifier
        if timestamp is None:
            timestamp = datetime.datetime.now()
        now_str = timestamp.strftime('%Y%m%d_%H%M')
        self._run_timestamp = now_str
        self._model = model
        self._misc = misc

    def run_timestamp(self):
        return self._run_timestamp

    def write(self, path):
        print(os.getcwd())
        fn = '%s_%s.meta' % (self.identifier, self._run_timestamp)
        with open(os.path.join(path, fn), 'wb') as w:
            _w = '''IDENTIFIER: %s\n''' % self.identifier
            w.write(_w.encode())
            _w = '''RUN: %s\n''' % self._run_timestamp
            w.write(_w.encode())
            if self._misc is not None:
                if isinstance(self._misc, dict):
                    _w = dict_to_str(self._misc)
                else:
                    _w = str(self._misc)
                _w = '''MISC: %s\n''' % _w
                w.write(_w.encode())
            _w = '''SUMMARY:\n'''
            w.write(_w.encode())
            self._model.summary(print_fn=lambda _: w.write((_ + '\n').encode()))
        return None

