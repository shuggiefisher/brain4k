import logging

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

        left_index = self.inputs[0].io.read_all([self.params['left_on']])
        left_index_flattened = left_index[self.params['left_on']].flatten()
        # create a minimal dataframe for the left part of the join
        left = pd.DataFrame({self.params['left_on']: left_index_flattened})

        right_keys = set([self.params['right_on']]) | set(self.params['retain_keys']['right'])
        right = self.inputs[1].io.read_all(keys=list(right_keys))

        df = pd.merge(
            left,
            right,
            how='left',
            left_on=self.params['left_on'],
            right_on=self.params['right_on']
        ).dropna()

        h5py_file = self.outputs[0].io.open(mode='w')
        self.outputs[0].io.create_dataset(
            h5py_file,
            self.params['output_keys'],
            df.shape[0]
        )

        # first copy the left side in chunks
        h5py_left = self.inputs[0].io.open()
        output_keys = {k: v for k, v in self.params['output_keys'].iteritems() if k in self.params['retain_keys']['left']}
        self.outputs[0].io.write_chunk(
            h5py_file,
            h5py_left,
            output_keys
        )
        self.outputs[0].io.close(h5py_left)

        # now copy the right side in from the merged dataframe
        output_keys = {k: v for k, v in self.params['output_keys'].iteritems() if k in self.params['retain_keys']['right']}
        self.outputs[0].io.write_chunk(
            h5py_file,
            df[output_keys.keys()],
            output_keys
        )
        self.outputs[0].io.save(h5py_file)

        logging.info(
            "Completed join saved as {0}".format(self.outputs[0].filename)
        )
