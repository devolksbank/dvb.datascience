import pytest

import pandas as pd

import dvb.datascience as ds
import dask.dataframe as dd


@pytest.mark.usefixtures("dataframe_engine")
class TestDump:

    @property
    def train_data(self):
        train_data = pd.DataFrame(
            [
                ["jan", 20, "M", 180],
                ["marie", 21, "W", 164],
                ["piet", 23, "M", 194],
                ["helen", 24, "W", 177],
                ["jan", 60, "U", 188],
            ],
            columns=["name", "age", "gender", "length"],
        ).sort_index(axis=1)

        if ds.pipeline.default_dataframe_engine == 'dask':
            train_data = dd.from_pandas(train_data, chunksize=1).compute()

        return train_data

    @pytest.fixture()
    def pipeline(self):
        return ds.Pipeline()

    @pytest.mark.skip_dataframe_engine("dask")
    def test_dump(self, pipeline):
        p = pipeline
        p.addPipe("read", ds.data.DataPipe(data=self.train_data))
        p.addPipe("dump", ds.eda.Dump(), [("read", "data", "df")])
        p.fit_transform()
        df_read = p.get_pipe_output("read")["data"]
        df_dump = p.get_pipe_output("dump")["output"]
        assert pd.DataFrame.equals(df_dump, df_read)
