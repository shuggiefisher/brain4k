import logging

from sklearn.cross_validation import train_test_split

from brain4k.transforms import PipelineStage


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
        if len(self.inputs) != 1:
            raise ValueError("TestTrainSplit expects just one input")
        if len(self.outputs) != 1:
            raise ValueError("TestTrainSplit expects just one output")

        logging.debug(
            "Reading input from {0} to split into test and training set"
            .format(self.inputs[0].filename)
        )

        h5py_input = self.inputs[0].io.open(mode='r')
        data_key = self.params['data']
        target_key = self.params['target']

        training_features, test_features, training_labels, test_labels = train_test_split(
            h5py_input[data_key],
            h5py_input[target_key],
            test_size=self.params['test_size']
        )
        data_dtype = h5py_input[data_key].dtype.char + h5py_input[data_key].dtype.name
        target_dtype = h5py_input[target_key].dtype.char + h5py_input[target_key].dtype.name
        output_keys = {
            'training_data': {
                'dtype':  data_dtype,
                'dimensions': training_features.shape[1]
            },
            'test_data': {
                'dtype':  data_dtype,
                'dimensions': test_features.shape[1]
            },
            'training_labels': {
                'dtype':  target_dtype,
                'dimensions': training_labels.shape[1]
            },
            'test_labels': {
                'dtype':  target_dtype,
                'dimensions': test_labels.shape[1]
            }
        }
        out = {
            'training_data': training_features,
            'test_data': test_features,
            'training_labels': training_labels,
            'test_labels': test_labels
        }
        h5py_output = self.outputs[0].io.open('w')
        self.outputs[0].io.write_chunk(h5py_output, out, output_keys)

        self.outputs[0].io.save(h5py_output)
        self.inputs[0].io.close(h5py_input)

