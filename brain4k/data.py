import os
import errno
import sys
import urllib

from data_interfaces import (
    CSVInterface,
    HDF5Interface,
    PickleInterface,
    FileInterface,
    MarkdownInterface,
    compute_file_hash
)


def path_to_file(repo_path, *args):
    path = os.path.join(repo_path, *args)

    return path


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Data(object):

    def __init__(self, name, config, data_config, *args, **kwargs):
        self.name = name
        self.filehash = data_config.get('sha1', None)
        local_filename = data_config.get('local_filename', None)
        url = data_config.get('url', None)
        self.data_type = data_config['data_type']
        self._set_filename(local_filename, url, config['repo_path'])
        self.io_class = FILE_INTERFACES[self.data_type]
        self.io = self.io_class(self.filename)

    def _set_filename(self, local_filename, url, repo_path):
        folders = FILE_PATHS.get(self.data_type, ['data', 'cache'])

        for folder in folders:
            self.filename = path_to_file(
                repo_path,
                folder,
                local_filename
            )

            # make sure the directory exists
            dirname = os.path.dirname(self.filename)
            mkdir_p(dirname)

            if os.path.exists(self.filename):
                break

        if url and not os.path.exists(self.filename):
            if not self.filehash:
                raise ValueError(
                    "A sha1 hash must be specified for the remote file {0}"\
                    .format(url)
                )
            download_with_progress_bar(url, self.filename)
            filehash = compute_file_hash(self.filename)
            if filehash != self.filehash:
                raise Exception(
                    "SHA1 hash of {0} does not match the one "
                    "specified in pipeline.json".format(self.filename)
                )


FILE_INTERFACES = {
    'hdf5': HDF5Interface,
    'csv': CSVInterface,
    'pickle': PickleInterface,
    'markdown': MarkdownInterface,
    'graph': FileInterface
}

FILE_PATHS = {
    'figure': [os.path.join('metrics', 'figures')],
    'markdown': ['metrics'],
}


def download_with_progress_bar(url, local_file):
    def dlProgress(count, block_size, total_size, url=url):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\rDownloading " + url + "...%d%%" % percent)
        sys.stdout.flush()

    urllib.urlretrieve(url, local_file, reporthook=dlProgress)
    sys.stdout.write("\n")