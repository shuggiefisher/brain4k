import logging
from copy import deepcopy

import pandas as pd

from brain4k.transforms import PipelineStage


class DataJoin(PipelineStage):
    """
    Perform a left join across two datasources.

    Useful in case one stage of the pipeline depends upon results generated
    at a prior stage, and the inputs have to be matched via an index
    """

    name = "org.brain4k.transforms.DataJoin"

    def join(self):
        if len(self.inputs) != 2:
            raise ValueError("Expecting two inputs to perform a join")
        if len(self.outputs) != 1:
            raise ValueError("Expecting one output for saving the join results")

        logging.info(
            "Starting join between {0} and {1} for {2}..."\
            .format(
                self.inputs[0].filename,
                self.inputs[1].filename,
                self.outputs[0].filename
            )
        )

        left_index = self.inputs[0].io.read_all([self.parameters['left_on']])
        left_index_flattened = left_index[self.parameters['left_on']].flatten()
        # create a minimal dataframe for the left part of the join
        left = pd.DataFrame({self.parameters['left_on']: left_index_flattened})

        right_keys = set([self.parameters['right_on']]) | set(self.parameters['retain_keys']['right'])
        right = self.inputs[1].io.read_all(keys=list(right_keys))

        df = pd.merge(
            left,
            right,
            how='left',
            left_on=self.parameters['left_on'],
            right_on=self.parameters['right_on']
        ).dropna()

        h5py_file = self.outputs[0].io.open(mode='w')
        h5py_left = self.inputs[0].io.open()

        left_output_keys = {k: v for k, v in self.parameters['output_keys'].iteritems() if k in self.parameters['retain_keys']['left']}
        right_output_keys = {k: v for k, v in self.parameters['output_keys'].iteritems() if k in self.parameters['retain_keys']['right']}
        for keyset, source in [(left_output_keys, h5py_left), (right_output_keys, df)]:
            for key in keyset:
                keyset[key]['shape'][0] = source[key].shape[0]

        self.outputs[0].io.create_dataset(
            h5py_file,
            self.parameters['output_keys'],
            deepcopy(left_output_keys).update(right_output_keys)
        )

        # first copy the left side in chunks
        self.outputs[0].io.write_chunk(
            h5py_file,
            h5py_left,
            left_output_keys
        )
        self.outputs[0].io.close(h5py_left)

        # now copy the right side in from the merged dataframe
        self.outputs[0].io.write_chunk(
            h5py_file,
            df[right_output_keys.keys()],
            right_output_keys
        )
        self.outputs[0].io.save(h5py_file)

        logging.info(
            "Completed join saved as {0}".format(self.outputs[0].filename)
        )
