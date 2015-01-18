import os
import hashlib
import json
from functools import partial

import h5py
import numpy as np
import pandas as pd


def compute_file_hash(file_path):
    with open(file_path, mode='rb') as f:
        d = hashlib.sha1()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)

        filehash = d.hexdigest()

    return filehash


def compute_json_hash(json_dict):
    json_str = json.dumps(json_dict)
    json_hash = hashlib.sha1().update(json_str).hexdigest()
    return json_hash


class HDF5Interface(object):

    def __init__(self, filename):
        self.filename = filename
        self.write_chunk_size = {}

    def open(self, mode='r'):
        h5py_file = h5py.File(self.filename, mode)
        return h5py_file

    def create_dataset(self, h5py_file, output_keys, rows):
        for key, params in output_keys.iteritems():
            h5py_file.create_dataset(
                key,
                [rows] + params['dimensions'],
                dtype=np.dtype(params['dtype']),
                chunks=True,
                compression="gzip",
                compression_opts=7
            )

    def save(self, h5py_file):
        h5py_file.close()

    def write_chunk(self, h5py_file, out):
        for key, values in out.iteritems():
            chunk_size = self.write_chunk_size.get(key, 50)
            for i in xrange(0, out.shape[0], chunk_size):
                last_index = i + chunk_size
                h5py_file[key][i:last_index] = out[key][i:last_index].reshape(self.params['dimensions'])


class CSVInterface(object):

    def __init__(self, filename):
        self.filename = filename

    def _get_compression(self):
        extension = os.path.basename(self.filename).split('.')
        if extension in ['gz', 'bz2']:
            return extension
        else:
            return None

    def read_chunk(self, chunk_size):
        df = pd.read_csv(
            self.filename,
            compression=self._get_compression(),
            usecols=['url'],
            chunksize=chunk_size
        )
        for chunk in df:
            yield chunk