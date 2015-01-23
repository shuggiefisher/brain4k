import logging

import pandas as pd

from brain4k.transforms import PipelineStage


class DataJoin(PipelineStage):
    """
    Perform a left join across two datasources.

    Useful in case one stage of the pipeline depends upon results generated
    at a prior stage, and these outputs have to be related to another input
    """

    name = "org.brain4k.transforms.DataJoin"

    def __init__(self, stage_config, config, **kwargs):
        super(DataJoin, self).__init__(self, stage_config, config, **kwargs)
        self.params = config['transforms'][stage_config['transform']]['params']

    def join(self):
        if len(self.inputs) != 2:
            raise ValueError("Expecting two inputs to perform a join")
        if len(self.outputs) != 1:
            raise ValueError("Expecting one output for saving the join results")

        logging.info(
            "Starting join between {0} and {1} for {2}..."\
            .format(
                self.input[0].filename,
                self.input[1].filename,
                self.output[0].filename
            )
        )

        left_keys = set([self.params['left_on']]) + set(self.params['retain_keys']['left'])
        left = self.inputs[0].io.read_chunk(chunksize=10000, keys=left_keys)

        right_keys = set([self.params['right_on']]) + set(self.params['retain_keys']['right'])
        right = self.inputs[1].io.read_chunk(chunksize=10000, keys=right_keys)

        df = pd.merge(left, right, how='left')

        h5py_file = self.output[0].io.open()
        self.output[0].io.create_dataset(
            h5py_file,
            self.params['output_keys'],
            df.shape[0]
        )
        self.output[0].io.save(h5py_file)

        logging.info("Completed join saved as {2}".format(self.output[0].filename))
