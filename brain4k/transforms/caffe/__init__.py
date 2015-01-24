import logging

import numpy as np
import caffe

from brain4k.transforms import PipelineStage


class BVLCCaffeNet(PipelineStage):

    name = "org.berkeleyvision.caffe.bvlc_caffenet"

    def predict(self):
        if len(self.outputs) != 1:
            raise ValueError("{0} expects only one output".format(self.name))

        for index, input_data in enumerate(self.inputs):
            h5py_file = self.outputs[index].io.open(mode='w')
            self.outputs[index].io.create_dataset(
                h5py_file,
                self.params['output_keys'],
                input_data.io.get_row_count()
            )

            chunk_size = 10

            for chunk_count, chunk in enumerate(input_data.io.read_chunk(chunk_size=chunk_size)):
                inputs, processed_urls = self._prepare_image_batch(chunk['url'], chunk_size)
                logging.debug("Making {0} predictions with {1}".format(chunk_size, self.name))
                layers_to_extract = list(set(self._net.blobs.keys()) & set(self.params['output_keys'].keys()))

                out = self._net.forward_all(
                    blobs=layers_to_extract,
                    **{self._net.inputs[0]: inputs}
                )
                for key in out.keys():
                    if key not in self.params['output_keys'].keys():
                        del out[key]
                    else:
                        if len(processed_urls) < chunk_size:
                            # get rid of padded zeros
                            out[key] = out[key][:len(processed_urls)]

                out['processed_urls'] = np.array(
                    processed_urls,
                    dtype=self.params['output_keys']['processed_urls']['dtype']
                )
                self.outputs[0].io.write_chunk(
                    h5py_file,
                    out,
                    self.params['output_keys'],
                    start_row=chunk_count*chunk_size
                )

            self.outputs[index].io.save(h5py_file)

    def _prepare_image_batch(self, urls, chunk_size):
        logging.debug("Fetching remote images...")
        images, processed_urls = self._fetch_images(urls)
        logging.debug("resizing images...")
        resized_images = [caffe.io.resize_image(im, self._net.image_dims) for im in images]

        inputs = np.zeros(
            (chunk_size, 3, self._net.image_dims[0], self._net.image_dims[1]),
            dtype=np.float32
        )
        logging.debug("preprocessing images...")
        for i, image in enumerate(resized_images):
            inputs[i] = self._net.preprocess(self._net.inputs[0], image)

        return inputs, processed_urls

    def _fetch_images(self, urls):
        images = []
        processed_urls = []
        for url in urls:
            try:
                image = caffe.io.load_image(url)
            except Exception:
                pass
            else:
                images.append(image)
                processed_urls.append(url)

        return images, processed_urls

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