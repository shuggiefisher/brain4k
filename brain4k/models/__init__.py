from brain4k.data import InputData, OutputData
from brain4k.data_interfaces import compute_json_hash, compute_file_hash


class PipelineStage(object):

    def __init__(self, stage_config, config):
        self.config = stage_config
        self.inputs = [InputData(input_name, config, config['data'][input_name]) for input_name in stage_config['inputs']]
        self.outputs = [OutputData(output_name, config, config['data'][output_name]) for output_name in stage_config['outputs']]

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

        for datum in data:
            if datum.name not in varying_data:
                filehash = compute_file_hash(datum.filename)
                stage_hashes.append(filehash)

        stage_hash = compute_json_hash({'stage_hashes': sorted(stage_hashes)})

        return stage_hash


MODELS = {
    "org.scikit-learn.naive_bayes.MultinomialNB": "brain4k.models.sklearn.NaiveBayes",
    "org.berkeleyvision.caffe.bvlc_caffenet": "brain4k.models.caffe.BVLCCaffeNet",
    "org.scikit-learn.cross_validation.test_train_split": "brain4k.models.sklearn.TestTrainSplit"
}
