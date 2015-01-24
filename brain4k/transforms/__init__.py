import os

from brain4k.data import Data, path_to_file
from brain4k.data_interfaces import compute_json_hash, compute_file_hash


class PipelineStage(object):

    def __init__(self, stage_config, config):
        self.config = stage_config
        self.inputs = [Data(name, config, config['data'][name]) for name in stage_config['inputs']]
        self.outputs = [Data(name, config, config['data'][name]) for name in stage_config['outputs']]
        self.params = config['transforms'][stage_config['transform']].get('params', {})

    def chain(self, actions):
        for action in actions:
            if not hasattr(self, action):
                raise ValueError(
                    "{0} does not support action {1}".format(self.name, action)
                )

        return [getattr(self, action)() for action in actions]

    def compute_hash(self):
        stage_hashes = []
        varying_data = self.config.get('accept_variance_in', [])
        data = self.inputs + self.files.values() + self.outputs

        non_varying_data = set([datum.filename for datum in data if datum.name not in varying_data])
        for datum in non_varying_data:
            filehash = compute_file_hash(datum.filename)
            stage_hashes.append(filehash)

        stage_hash = compute_json_hash({'stage_hashes': sorted(stage_hashes)})

        return stage_hash


TRANSFORMS = {
    "org.scikit-learn.naive_bayes.MultinomialNB": "brain4k.transforms.sklearn.NaiveBayes",
    "org.berkeleyvision.caffe.bvlc_caffenet": "brain4k.transforms.caffe.BVLCCaffeNet",
    "org.scikit-learn.cross_validation.test_train_split": "brain4k.transforms.sklearn.TestTrainSplit",
    "org.brain4k.transforms.data_join": "brain4k.transforms.sklearn.DataJoin",
    "org.scikit-learn.metrics.confusion_matrix": "brain4k.transforms.sklearn.metrics.ConfusionMatrix",
}


def render_metrics(config):
    """
    Concatenate the markdown files that make up the metrics.
    Output it as the README.md
    """
    input_files = []
    output_file = path_to_file(config['repo_path'], 'README.md')

    header_file = path_to_file(
        config['repo_path'],
        os.path.join('metrics', 'HEADER.md')
    )
    if os.path.exists(header_file):
        input_files.append(header_file)

    for metric_name in config['metrics']:
        metric_file = os.path.join(
            'metrics',
            config['data'][metric_name]['filename']
        )
        input_files.append(metric_file)

    with open(output_file, 'w') as outfile:
        for fname in input_files:
            with open(fname) as infile:
                outfile.write(infile.read())
