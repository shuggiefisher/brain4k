import os
import logging


class PipelineStage(object):

    def __init__(self, stage_config, config):
        from brain4k.data import Data

        self.config = stage_config
        self.inputs = [Data(name, config, config['data'][name]) for name in stage_config['inputs']]
        self.outputs = [Data(name, config, config['data'][name]) for name in stage_config['outputs']]
        self.transform_name = stage_config['transform']
        self.parameters = config['transforms'][self.transform_name].get('parameters', {})
        files = config['transforms'][self.transform_name].get('files', {})
        self.files = {name: Data(data_name, config, config['data'][data_name]) for name, data_name in files.iteritems()}

    def chain(self, actions):
        for action in actions:
            if not hasattr(self, action):
                raise ValueError(
                    "{0} does not support action {1}".format(self.name, action)
                )

        try:
            results = [getattr(self, action)() for action in actions]
        except Exception as e:
            logging.exception(
                "Encountered unhandled exception during when"
                " calling {0} on {1}: {2}"
                .format(action, self.name, e)
            )
            logging.exception(
                "Deleting all output blobs for stage"
            )
            for datum in self.outputs:
                os.remove(datum.filename)
            raise
        else:
            return results

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

    def blob_files_exist(self):
        """
        Before computing the sha1 hash, we might want to check that the
        blob files all exist
        """
        data = set(self.inputs) | set(self.files.values()) | set(self.outputs)
        for datum in data:
            if not os.path.exists(datum.filename):
                return False

        return True



TRANSFORMS = {
    "org.scikit-learn.naive_bayes.MultinomialNB": "brain4k.transforms.sklearn.NaiveBayes",
    "org.berkeleyvision.caffe.bvlc_caffenet": "brain4k.transforms.caffe.BVLCCaffeNet",
    "org.scikit-learn.cross_validation.test_train_split": "brain4k.transforms.sklearn.TestTrainSplit",
    "com.brain4k.transforms.data_join": "brain4k.transforms.b4k.DataJoin",
    "org.scikit-learn.metrics.confusion_matrix": "brain4k.transforms.sklearn.metrics.ConfusionMatrix",
}
