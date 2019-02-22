import pytest

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import dvb.datascience as ds
import dask.dataframe as dd


@pytest.mark.usefixtures("dataframe_engine")
class TestPredictor:
    @property
    def train_data(self):
        train_data = pd.DataFrame(
            [[0, 0], [1, 0], [2, 1], [3, 1]],
            columns=["X", "y"],
            index=["a", "b", "c", "d"],
        )

        if ds.pipeline.default_dataframe_engine == 'dask':
            train_data = dd.from_pandas(train_data, chunksize=1).compute()

        return train_data

    @property
    def test_data(self):
        test_data = pd.DataFrame(
            [[0], [1.5], [2.5], [6]], columns=["X"], index=self.train_data.index
        )

        if ds.pipeline.default_dataframe_engine == 'dask':
            test_data = dd.from_pandas(test_data, chunksize=1).compute()

        return test_data

    def get_pipeline(self):
        p = ds.Pipeline()
        p.addPipe("read", ds.data.DataPipe())
        p.addPipe("metadata", ds.data.DataPipe(data={"y_true_label": "y"}))
        p.addPipe(
            "clf",
            ds.predictor.SklearnClassifier(clf=KNeighborsClassifier, n_neighbors=3),
            [("read", "data", "df"), ("metadata", "data", "df_metadata")],
        )

        return p

    def test_predict(self):
        p = self.get_pipeline()

        params = {"read": {"data": self.train_data}, "clf": {}}

        p.fit_transform(transform_params=params, fit_params=params)
        assert list(p.get_pipe_output("clf")["predict"].index) == ["a", "b", "c", "d"]
        assert set(p.get_pipe_output("clf")["predict"].columns) == \
            set(["y", "y_pred_0", "y_pred_1", "y_pred"])
        assert p.get_pipe_output("clf")["predict"].iloc[0]["y_pred"] == 0
        assert p.get_pipe_output("clf")["predict"].iloc[3]["y_pred"] == 1

        params["read"]["data"] = self.test_data
        p.transform(transform_params=params)
        assert list(p.get_pipe_output("clf")["predict"].index) == ["a", "b", "c", "d"]
        assert set(p.get_pipe_output("clf")["predict"].columns) == \
            set(["y_pred_0", "y_pred_1", "y_pred"])
        assert p.get_pipe_output("clf")["predict"].iloc[0]["y_pred"] == 0
        assert p.get_pipe_output("clf")["predict"].iloc[3]["y_pred"] == 1
