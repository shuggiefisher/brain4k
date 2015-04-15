import logging
from collections import defaultdict

import numpy as np
import caffe

from brain4k.transforms import PipelineStage
from brain4k.transforms.b4k import grouper


class BVLCCaffeNet(PipelineStage):

    name = "org.berkeleyvision.caffe.bvlc_caffenet"

    def predict_for_url(self):
        """
        has been written to support multiple urls, but in practice
        command-line only expects 1 url.  This could be re-written to do only
        """
        urls = [self.inputs[0].value]
        chunk_size = 10
        results = defaultdict(list)

        for urls_chunk in grouper(chunk_size, urls):
            inputs, processed_urls = self._prepare_image_batch(urls_chunk, chunk_size)
            unprocessed_urls = set(urls) - set(processed_urls)
            if unprocessed_urls:
                logging.warning(
                    "some urls: {0} were not fetched successfully"
                    .format(unprocessed_urls)
                )
            if processed_urls:
                out = self._extract_features(inputs, processed_urls, chunk_size)
                for key, values in out.iteritems():
                    output_shape = [values.shape[0]] + list(self.parameters['output_keys'][key]['shape'][1:])
                    results[key].append(values.reshape(output_shape))

        for key, values in results.iteritems():
            results[key] = np.concatenate(values)

        return [results]


    def predict(self):
        if len(self.outputs) != 1:
            raise ValueError("{0} expects only one output".format(self.name))

        for index, input_data in enumerate(self.inputs):
            h5py_file = self.outputs[index].io.open(mode='w')
            self.outputs[index].io.create_dataset(
                h5py_file,
                self.parameters['output_keys'],
                input_data.io.get_row_count()
            )

            chunk_size = 10

            for chunk_count, chunk in enumerate(input_data.io.read_chunk(chunk_size=chunk_size)):
                inputs, processed_urls = self._prepare_image_batch(chunk['url'], chunk_size)
                if len(processed_urls) == 0:
                    logging.warning(
                        "No images were successfully fetched from urls: {0}"
                        .format(chunk['url'])
                    )
                else:
                    out = self._extract_features(inputs, processed_urls, chunk_size)

                    self.outputs[0].io.write_chunk(
                        h5py_file,
                        out,
                        self.parameters['output_keys'],
                        start_row=chunk_count*chunk_size
                    )

            # TODO: shrink dataset size to remove zeros
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
            if self.parameters.get('gpu', False):
                self._caffe_net.set_gpu()

        return self._caffe_net

    def _extract_features(self, inputs, processed_urls, chunk_size):
        logging.debug("Making {0} predictions with {1}".format(chunk_size, self.name))
        layers_to_extract = list(set(self._net.blobs.keys()) & set(self.parameters['output_keys'].keys()))

        out = self._net.forward_all(
            blobs=layers_to_extract,
            **{self._net.inputs[0]: inputs}
        )
        for key in out.keys():
            if key not in self.parameters['output_keys'].keys():
                del out[key]
            else:
                if len(processed_urls) < chunk_size:
                    # get rid of padded zeros
                    out[key] = out[key][:len(processed_urls)]

        out['processed_urls'] = np.array(
            processed_urls,
            dtype=self.parameters['output_keys']['processed_urls']['dtype']
        )

        return out