from brain4k.models import PipelineStage


class NaiveBayes(PipelineStage):

    name = "org.scikit-learn.naive_bayes.MultinomialNB"

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()


class TestTrainSplit(PipelineStage):

    name = "org.scikit-learn.cross_validation.test_train_split"

    def __init__(self, config):
        super(TestTrainSplit, self).__init__(config)
        self.data = config['data']
        self.target = config['target']
        self.ratio = config['ratio']

    def split(self):
        raise NotImplementedError()
