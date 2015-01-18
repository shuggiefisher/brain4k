import itertools

from data import InputData, OutputData


class PipelineStage(object):

    def __init__(self, stage_config, config):
        self.config = stage_config
        self.inputs = [InputData(input_name, config['data'][input_name]) for input_name in stage_config['inputs']]
        self.outputs = [OutputData(output_name, config['data'][output_name]) for output_name in stage_config['outputs']]

    def chain(self, actions):
        for action in actions:
            if not hasattr(self, action):
                raise ValueError(
                    "{0} does not support action {1}".format(self.name, action)
                )

        return itertools.chain(*(getattr(self, action)() for action in actions))


MODELS = {
    "org.scikit-learn.naive_bayes.MultinomialNB": "models.sklearn.NaiveBayes",
    "org.berkeleyvision.caffe.bvlc_caffenet": "models.caffe.BVLCCaffeNet",
    "org.scikit-learn.cross_validation.test_train_split": "models.sklearn.TestTrainSplit"
}
