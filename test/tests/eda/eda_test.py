import pytest

import pandas as pd

import dvb.datascience as ds
import dask.dataframe as dd


@pytest.mark.usefixtures("dataframe_engine")
class TestEDA:

    @property
    def train_data(self):
        train_data = pd.DataFrame(
            [
                ["jan", 20, 0, 180],
                ["marie", 21, 1, 164],
                ["piet", 23, 0, 194],
                ["helen", 24, 1, 177],
                ["jan", 60, 0, 188],
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
    def test_logit(self, pipeline):
        p = pipeline
        p.addPipe("read", ds.data.DataPipe(data=self.train_data))
        p.addPipe("metadata", ds.data.DataPipe(key="df_metadata", data={'classes': [0, 1], 'y_true_label': 'gender', 'X_labels': ['age', 'length']}))
        p.addPipe("logit", ds.eda.LogitSummary(), [("read", "data", "df"), ("metadata", "df_metadata", "df_metadata")])
        p.fit_transform()
        summary = p.get_pipe_output("logit")["summary"]
        assert 'Logit Regression Results' in summary.as_csv()
