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
        self.data_type = data_config.get('data_type', '')

        self.filehash = data_config.get('sha1', None)
        local_filename = data_config.get('local_filename', None)
        url = data_config.get('url', None)

        if self.data_type != 'argument':
            if not local_filename and not url:
                raise ValueError(
                    "Each Data blob must have a local_filename or a url and sha1"
                    " hash specified."
                )

            self._set_filename(local_filename, url, config['repo_path'])
            self.io_class = FILE_INTERFACES.get(self.data_type, FileInterface)
            self.io = self.io_class(self.filename)

    def _set_filename(self, local_filename, url, repo_path):
        folders = FILE_PATHS.get(self.data_type, ['data', 'cache'])

        if local_filename:
            base_name = os.path.basename(local_filename)
        else:
            base_name = os.path.basename(url)

        for folder in folders:
            self.filename = path_to_file(
                repo_path,
                folder,
                base_name
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
    'markdown': MarkdownInterface
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