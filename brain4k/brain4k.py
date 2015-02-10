import os
import logging
from argparse import ArgumentParser

from pipeline import execute_pipeline


logging.basicConfig(level=logging.DEBUG)


class Brain4kArgumentParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(Brain4kArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument(
            'repo path',
            nargs='?',
            default=os.getcwd(),
            help='Path to the brain4k repository'
        )
        self.add_argument(
            '--force-render-metrics',
            dest='force_render_metrics',
            action='store_false',
            help='Re-render the metrics and README.md'
        )


def run():
    parser = Brain4kArgumentParser()
    brain4k_args = parser.parse_args()

    repo_path = getattr(brain4k_args, 'repo path')
    if not os.path.isabs(repo_path):
        repo_path = os.path.join(os.getcwd(), repo_path)

    execute_pipeline(
        repo_path,
        force_render_metrics=brain4k_args.force_render_metrics
    )
