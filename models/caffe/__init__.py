import numpy as np
import caffe

from models import PipelineStage
from data import Data


class BVLCCaffeNet(PipelineStage):

    name = "org.berkeleyvision.caffe.bvlc_caffenet"

    def __init__(self, stage_config, config, **kwargs):
        self.params = config['model'][stage_config['model']]['params']
        self.files = [Data(name, data_config) for name, data_config in config['model'][stage_config['model']]['files'].iteritems()]
        super(BVLCCaffeNet, self).__init(stage_config, config, **kwargs)
        if len(self.outputs) != 1:
            raise ValueError("{0} expects only one output".format(self.name))

    def predict(self):

        for index, input_data in enumerate(self.inputs):
            h5py_file = self.outputs[index].writer.open()
            self.outputs[index].create_dataset(
                h5py_file,
                self.params['output_keys'],
                input_data.shape[0]
            )

            chunk_size = 10

            for chunk in input_data.read_chunk(chunk_size=chunk_size):
                inputs = self._prepare_image_batch(chunk['url'], chunk_size)
                out = self._net.forward_all(
                    blobs=self.params['output_keys'].keys(),
                    **{self._net.inputs[0]: inputs}
                )
                self.outputs[0].writer.write_chunk(h5py_file, out)

            self.outputs[index].writer.save(h5py_file)

    def _prepare_image_batch(self, urls, chunk_size):
        images = [caffe.io.load_image(url) for url in urls]
        resized_images = [caffe.io.resize_image(im, self._net.image_dims) for im in images]

        inputs = np.zeros([chunk_size] + self._net.image_dims + [3], dtype=np.float32)
        for i, image in enumerate(resized_images):
            inputs[i] = self._net.preprocess(self._net.inputs[0], image)

        return inputs

    @property
    def _net(self):
        if not hasattr(self._caffe_net):
            self._caffe_net = caffe.Classifier(
                self.files['prototxt'],
                self.files['weights'],
                mean=self.files['mean'],
                channel_swap=(2,1,0),
                raw_scale=255
            )
            if self.params.get('gpu', False):
                self._caffe_net.set_gpu()

        return self._caffe_net