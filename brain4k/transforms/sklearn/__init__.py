import logging

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

from brain4k.transforms import PipelineStage


class NaiveBayes(PipelineStage):

    name = "org.scikit-learn.naive_bayes.MultinomialNB"

    def train(self):
        logging.debug(
            "Reading training data and target from {0} for {1}..."
            .format(self.inputs[0].filename, self.name)
        )
        h5py_input = self.inputs[0].io.open(mode='r')
        data = h5py_input['training_data']
        target = h5py_input['training_target'].value.flatten()

        logging.debug("Fitting {0} to data...".format(self.name))
        self.estimator = MultinomialNB()
        self.estimator.fit(data, target)

        self.inputs[0].io.close(h5py_input)
        self.outputs[0].io.save(self.estimator)

    def test(self):
        logging.debug(
            "Reading testing data and target from {0} for {1}..."
            .format(self.inputs[0].filename, self.name)
        )
        h5py_input = self.inputs[0].io.open(mode='r')
        data = h5py_input['test_data']
        target = h5py_input['test_target'].value.flatten()

        logging.debug("Testing {0}...".format(self.name))

        # if the train process has not just been run, the estimator
        # should be loaded as an input
        predictions = self.estimator.predict(data)

        self.inputs[0].io.close(h5py_input)

        h5py_output = self.outputs[1].io.open('w')
        out = {
            "predictions": predictions,
            "actual": target
        }
        output_keys = {
            'predictions': {
                'dtype': predictions.dtype.name,
                'shape': predictions.shape
            },
            'actual': {
                'dtype': target.dtype.name,
                'shape': target.shape
            }
        }
        self.outputs[1].io.create_dataset(
            h5py_output,
            output_keys
        )
        self.outputs[1].io.write_chunk(
            h5py_output,
            out,
            output_keys
        )

        self.outputs[1].io.save(h5py_output)

    def predict(self):
        features = self.inputs[0].value[self.parameters['data']]
        self.estimator = self.inputs[1].read_all()
        predicted_labels = self.estimator.predict(features)
        for label, url in zip(predicted_labels, self.inputs[0].value['processed_url']):
            print "{0} : {1}".format(label, url)


class TestTrainSplit(PipelineStage):

    name = "org.scikit-learn.cross_validation.test_train_split"

    def split(self):
        if len(self.inputs) != 1:
            raise ValueError("{0} expects just one input".format(self.name))
        if len(self.outputs) != 1:
            raise ValueError("{0} expects just one output".format(self.name))

        logging.debug(
            "Reading input from {0} to split into test and training set"
            .format(self.inputs[0].filename)
        )

        h5py_input = self.inputs[0].io.open(mode='r')
        data_key = self.parameters['data']
        target_key = self.parameters['target']

        training_features, test_features, training_labels, test_labels = train_test_split(
            h5py_input[data_key],
            h5py_input[target_key],
            test_size=self.parameters['test_size']
        )
        data_dtype = h5py_input[data_key].dtype.name
        target_dtype = h5py_input[target_key].dtype.name
        output_keys = {
            'training_data': {
                'dtype':  data_dtype,
                'shape': training_features.shape
            },
            'test_data': {
                'dtype':  data_dtype,
                'shape': test_features.shape
            },
            'training_target': {
                'dtype':  target_dtype,
                'shape': training_labels.shape
            },
            'test_target': {
                'dtype':  target_dtype,
                'shape': test_labels.shape
            }
        }
        out = {
            'training_data': training_features,
            'test_data': test_features,
            'training_target': training_labels,
            'test_target': test_labels
        }
        h5py_output = self.outputs[0].io.open('w')
        self.outputs[0].io.create_dataset(
            h5py_output,
            output_keys
        )
        self.outputs[0].io.write_chunk(h5py_output, out, output_keys)

        self.outputs[0].io.save(h5py_output)
        self.inputs[0].io.close(h5py_input)

