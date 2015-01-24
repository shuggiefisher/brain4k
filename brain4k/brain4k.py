import os
import logging
from argparse import ArgumentParser

from pipeline import execute_pipeline


logging.basicConfig(level=logging.DEBUG)


class Brain4kArgumentParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(Brain4kArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument('repo path', nargs='?', default=os.getcwd())


def run():
    parser = Brain4kArgumentParser()
    brain4k_args = parser.parse_args()

    repo_path = getattr(brain4k_args, 'repo path')
    if not os.path.isabs(repo_path):
        repo_path = os.path.join(os.getcwd(), repo_path)

    execute_pipeline(repo_path)
