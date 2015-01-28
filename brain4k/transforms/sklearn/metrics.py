from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from brain4k.transforms import PipelineStage


class ConfusionMatrix(PipelineStage):

    def plot(self):
        if not 0 < len(self.inputs) < 3:
            raise ValueError("{0} expects just one or two inputs".format(self.name))
        if len(self.outputs) != 2:
            raise ValueError("{0} expects two outputs".format(self.name))

        input_keys = self.parameters.get('input_keys', {})
        predictions_key = input_keys.get('predictions', 'predictions')
        actual_key = input_keys.get('actual', 'actual')

        h5py_input = self.inputs[0].io.open(mode='r')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # get labels from csv
        label_names = None
        if len(self.inputs) > 1:
            label_df = self.inputs[1].io.read_all(index_col=0)
            label_df.index = label_df.index.astype(h5py_input[actual_key].dtype.name)
            max_label_known = max(
                h5py_input[predictions_key].value.max(),
                h5py_input[actual_key].value.max()
            )
            label_names = list(label_df.name.values[:max_label_known+1])
            ax.set_xticklabels([''] + label_names)
            ax.set_yticklabels([''] + label_names)

        confusion = confusion_matrix(
            h5py_input[actual_key].value,
            h5py_input[predictions_key].value,
            range(max_label_known+1)
        )
        cax = ax.matshow(confusion)

        # write scores in squares

        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig.colorbar(cax)

        # render plot to image output
        fig.savefig(self.outputs[0].filename, transparent=True)

        # write markdown fragment as output
        self.outputs[1].io.write({
            'confusion_matrix': confusion,
            'image_src': self.outputs[0].filename
        })
