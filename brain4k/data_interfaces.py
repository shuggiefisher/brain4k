import os
import hashlib
import json
import logging
import cPickle
from functools import partial

import h5py
import numpy as np
import pandas as pd


def compute_file_hash(file_path):
    logging.debug("computing hash for {0}".format(file_path))
    with open(file_path, mode='rb') as f:
        d = hashlib.sha1()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)

        filehash = d.hexdigest()

    return filehash


def compute_json_hash(json_dict):
    json_str = json.dumps(json_dict)
    json_hash = hashlib.sha1()
    json_hash.update(json_str)
    return json_hash.hexdigest()


class FileInterface(object):

    def __init__(self, filename):
        self.filename = filename


class HDF5Interface(FileInterface):

    def __init__(self, filename):
        super(HDF5Interface, self).__init__(filename)
        self.write_chunk_size = {}

    def open(self, mode='r'):
        h5py_file = h5py.File(self.filename, mode)
        return h5py_file

    def create_dataset(self, h5py_file, output_keys, rows):
        logging.info("Creating HDF5 dataset {0}".format(self.filename))
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

    def read_all(self, keys):
        h5py_file = self.open()
        contents = {key: h5py_file[key].value for key in keys}
        self.close(h5py_file)
        return contents

    def write_chunk(self, h5py_file, out, output_keys, start_row=0):
        for key in output_keys.keys():
            chunk_size = min(self.write_chunk_size.get(key, 500), out[key].shape[0])
            output_shape = [chunk_size]
            if output_keys[key]['dimensions'] > 1:
                output_shape += output_keys[key]['dimensions']
            for i in xrange(start_row, start_row + out[key].shape[0], chunk_size):
                last_index = i + chunk_size
                last_index_out = min(last_index - start_row, out[key].shape[0])
                h5py_file[key][i:last_index] = out[key][i-start_row:last_index_out].reshape(output_shape)

    def close(self, h5py_file):
        h5py_file.close()


class CSVInterface(FileInterface):

    def get_row_count(self):
        row_count = sum(1 for line in open(self.filename))
        return row_count - 1

    def _get_compression(self):
        extension = os.path.basename(self.filename).split('.')
        if extension in ['gz', 'bz2']:
            return extension
        else:
            return None

    def read_chunk(self, chunk_size, keys=['url']):
        df = pd.read_csv(
            self.filename,
            compression=self._get_compression(),
            usecols=keys,
            chunksize=chunk_size
        )
        for chunk in df:
            yield chunk

    def read_all(self, keys=['url']):
        df = pd.read_csv(
            self.filename,
            compression=self._get_compression(),
            usecols=keys
        )
        return df


class PickleInterface(FileInterface):

    def save(self, obj):
        with open(self.filename, 'rb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


class MarkdownInterface(FileInterface):

    def write(self, context):
        """
        TODO: accept templates as input
        """
        with open(self.filename, 'w') as f:
            image = "![Confusion Matrix Caption][{0}]\n\n".format(context['image_src'])
            values = "Confusion matrix:\n{0}".format(context['confusion_matrix'])
            f.write(image + values)

