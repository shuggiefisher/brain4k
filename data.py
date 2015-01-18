import os

from data_interfaces import CSVInterface, HDF5Interface


def path_to_file(repo_path, *args):
    path = os.path.join(repo_path, *args)

    return path


class Data(object):

    def __init__(self, name, config, data_config, folder='cache'):
        self.name = name
        self.filehash = data_config.get('sha1', None)
        local_filename = data_config.get('local_filename', None)
        if local_filename:
            self.filename = path_to_file(
                config['repo_path'],
                folder,
                local_filename
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