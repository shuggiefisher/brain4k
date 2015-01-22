import logging

import numpy as np
import caffe

from brain4k.models import PipelineStage
from brain4k.data import Data


class BVLCCaffeNet(PipelineStage):

    name = "org.berkeleyvision.caffe.bvlc_caffenet"

    def __init__(self, stage_config, config, **kwargs):
        self.params = config['models'][stage_config['model']]['params']
        self.files = {name: Data(data_name, config, config['data'][data_name]) for name, data_name in config['models'][stage_config['model']]['files'].iteritems()}
        super(BVLCCaffeNet, self).__init__(stage_config, config, **kwargs)
        if len(self.outputs) != 1:
            raise ValueError("{0} expects only one output".format(self.name))

    def predict(self):

        for index, input_data in enumerate(self.inputs):
            h5py_file = self.outputs[index].writer.open(mode='w')
            self.outputs[index].writer.create_dataset(
                h5py_file,
                self.params['output_keys'],
                input_data.reader.get_row_count()
            )

            chunk_size = 10

            for chunk in input_data.reader.read_chunk(chunk_size=chunk_size):
                inputs = self._prepare_image_batch(chunk['url'], chunk_size)
                logging.debug("Making {0} predictions with {1}".format(chunk_size, self.name))
                out = self._net.forward_all(
                    blobs=self.params['output_keys'].keys(),
                    **{self._net.inputs[0]: inputs}
                )
                for key in out.keys():
                    if key not in self.params['output_keys'].keys():
                        del out[key]
                self.outputs[0].writer.write_chunk(h5py_file, out, self.params['output_keys'])

            self.outputs[index].writer.save(h5py_file)

    def _prepare_image_batch(self, urls, chunk_size):
        logging.debug("Fetching remote images...")
        images = [caffe.io.load_image(url) for url in urls]
        logging.debug("resizing images...")
        resized_images = [caffe.io.resize_image(im, self._net.image_dims) for im in images]

        inputs = np.zeros(
            (chunk_size, 3, self._net.image_dims[0], self._net.image_dims[1]),
            dtype=np.float32
        )
        logging.debug("preprocessing images...")
        for i, image in enumerate(resized_images):
            inputs[i] = self._net.preprocess(self._net.inputs[0], image)

        return inputs

    @property
    def _net(self):
        if not hasattr(self, '_caffe_net'):
            logging.debug("Initializing Caffe network")
            self._caffe_net = caffe.Classifier(
                str(self.files['prototxt'].filename),
                str(self.files['weights'].filename),
                mean=np.load(self.files['mean'].filename),
                channel_swap=(2,1,0),
                raw_scale=255
            )
            if self.params.get('gpu', False):
                self._caffe_net.set_gpu()

        return self._caffe_net