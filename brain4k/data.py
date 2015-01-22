import os
import sys
import urllib

from brain4k.data_interfaces import CSVInterface, HDF5Interface, compute_file_hash


def path_to_file(repo_path, *args):
    path = os.path.join(repo_path, *args)

    return path


class Data(object):

    def __init__(self, name, config, data_config, folder='cache'):
        self.name = name
        self.filehash = data_config.get('sha1', None)
        local_filename = data_config.get('local_filename', None)
        url = data_config.get('url', None)
        self._set_filename(local_filename, url, config['repo_path'], folder)

    def _set_filename(self, local_filename, url, repo_path, folder):
        if local_filename:
            self.filename = path_to_file(
                repo_path,
                folder,
                local_filename
            )
        elif url:
            if not self.filehash:
                raise ValueError(
                    "A sha1 hash must be specified for the remote file {0}"\
                    .format(url)
                )
            self.filename = path_to_file(
                repo_path,
                'cache',
                os.path.basename(url)
            )
            if not os.path.exists(self.filename):
                download_with_progress_bar(url, self.filename)
            filehash = compute_file_hash(self.filename)
            if filehash != self.filehash:
                raise Exception(
                    "SHA1 hash of {0} does not match the one "
                    "specified in pipeline.json".format(self.filename)
                )


class InputData(Data):

    def __init__(self, name, config, data_config, folder='data'):
        super(InputData, self).__init__(name, config, data_config, folder)
        self.reader_class = READERS[data_config['data_type']]
        self.reader = self.reader_class(self.filename)

    def read_chunk(self, chunk_size=1000):
        return self.reader.read_chunk(chunk_size)


class OutputData(Data):

    def __init__(self, name, config, data_config, folder='cache'):
        super(OutputData, self).__init__(name, config, data_config, folder)
        self.writer_class = READERS[data_config['data_type']]
        self.writer = self.writer_class(self.filename)

    def open(self, mode='r'):
        raise NotImplementedError()

    def save(self, filehandle):
        raise NotImplementedError()


READERS = {
    'hdf5': HDF5Interface,
    'csv': CSVInterface
}


def download_with_progress_bar(url, local_file):
    def dlProgress(count, block_size, total_size, url=url):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\rDownloading " + url + "...%d%%" % percent)
        sys.stdout.flush()

    urllib.urlretrieve(url, local_file, reporthook=dlProgress)
    sys.stdout.write("\n")