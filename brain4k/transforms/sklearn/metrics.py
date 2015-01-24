import logging

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from brain4k.transforms import PipelineStage


class ConfusionMatrix(PipelineStage):

    def plot(self):

        input_keys = self.params.get('input_keys', {})
        predictions_key = input_keys.get('predictions', 'predictions')
        actual_key = input_keys.get('actual', 'actual')

        h5py_input = self.inputs[0].io.open(mode='r')

        confusion = confusion_matrix(
            h5py_input[actual_key].value,
            h5py_input[predictions_key].value
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(confusion)

        # get labels from csv
        # ax.set_xticklabels(['', 'cat', 'dog'])
        # ax.set_yticklabels(['', 'cat', 'dog'])

        # write scores in squares

        # set background to transparent

        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # render plot to image output
        # plt.show()

        # write markdown fragment as output
