import os

from data_interfaces import CSVInterface, HDF5Interface, compute_file_hash


def path_to_file(repo_path, *args):
    path = os.path.join(repo_path, *args)

    return path


class Data(object):

    def __init__(self, name, config, folder='cache'):
        self.name = name
        self.filehash = config.get('sha1', None)
        local_filename = config.get('local_filename', None)
        if local_filename:
            self.filename = path_to_file(
                config['repo_path'],
                [folder, local_filename],
            )

    def matches_hash(self):
        """
        Does the hash in pipeline json, match the hash of the cached file?
        """
        if os.path.exists(self.filename):
            filehash = compute_file_hash(self.filename)
            if self.filehash == filehash:
                return True

        return False


class InputData(Data):

    def __init__(self, name, config, folder='data'):
        super(InputData, self).__init__(name, config, folder)
        self.reader_class = READERS[config['input_type']]
        self.reader = self.reader_class(self.filename)

    def read_chunk(self, chunk_size=1000):
        return self.reader.read_chunk(chunk_size)


class OutputData(Data):

    def __init__(self, name, config, folder='cache'):
        super(InputData, self).__init__(name, config, folder)
        self.writer_class = READERS[config['input_type']]
        self.writer = self.writer_class(self.filename)

    def open(self, mode='r'):
        raise NotImplementedError()

    def save(self, filehandle):
        raise NotImplementedError()


READERS = {
    'hdf5': HDF5Interface,
    'csv': CSVInterface
}