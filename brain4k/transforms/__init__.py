
class PipelineStage(object):

    def __init__(self, stage_config, config):
        from brain4k.data import Data

        self.config = stage_config
        self.inputs = [Data(name, config, config['data'][name]) for name in stage_config['inputs']]
        self.outputs = [Data(name, config, config['data'][name]) for name in stage_config['outputs']]
        self.params = config['transforms'][stage_config['transform']].get('params', {})
        files = config['transforms'][stage_config['transform']].get('files', {})
        self.files = {name: Data(data_name, config, config['data'][data_name]) for name, data_name in files.iteritems()}

    def chain(self, actions):
        for action in actions:
            if not hasattr(self, action):
                raise ValueError(
                    "{0} does not support action {1}".format(self.name, action)
                )

        return [getattr(self, action)() for action in actions]

    def compute_hash(self):
        from brain4k.data_interfaces import compute_json_hash, compute_file_hash

        stage_hashes = []
        varying_data = self.config.get('accept_variance_in', [])
        data = self.inputs + self.files.values() + self.outputs

        non_varying_data = set([datum.filename for datum in data if datum.name not in varying_data])
        for filename in non_varying_data:
            filehash = compute_file_hash(filename)
            stage_hashes.append(filehash)

        stage_hash = compute_json_hash({'stage_hashes': sorted(stage_hashes)})

        return stage_hash


TRANSFORMS = {
    "org.scikit-learn.naive_bayes.MultinomialNB": "brain4k.transforms.sklearn.NaiveBayes",
    "org.berkeleyvision.caffe.bvlc_caffenet": "brain4k.transforms.caffe.BVLCCaffeNet",
    "org.scikit-learn.cross_validation.test_train_split": "brain4k.transforms.sklearn.TestTrainSplit",
    "com.brain4k.transforms.data_join": "brain4k.transforms.b4k.DataJoin",
    "org.scikit-learn.metrics.confusion_matrix": "brain4k.transforms.sklearn.metrics.ConfusionMatrix",
}
