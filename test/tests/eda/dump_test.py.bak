import unittest

import pandas as pd

import dvb.datascience as ds


class TestDump(unittest.TestCase):
    def setUp(self):
        self.pipeline = ds.Pipeline()

        self.train_data = pd.DataFrame(
            [
                ["jan", 20, "M", 180],
                ["marie", 21, "W", 164],
                ["piet", 23, "M", 194],
                ["helen", 24, "W", 177],
                ["jan", 60, "U", 188],
            ],
            columns=["name", "age", "gender", "length"],
        )

    def test_dump(self):
        p = self.pipeline
        p.addPipe("read", ds.data.DataPipe(data=self.train_data))
        p.addPipe("dump", ds.eda.Dump(), [("read", "data", "df")])
        p.fit_transform()
        df_read = p.get_pipe_output("read")["data"]
        df_dump = p.get_pipe_output("dump")["output"]
        self.assertTrue(pd.DataFrame.equals(df_dump, df_read))
