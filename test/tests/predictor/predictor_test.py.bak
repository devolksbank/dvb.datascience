import unittest

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

import dvb.datascience as ds


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.train_data = pd.DataFrame(
            [[0, 0], [1, 0], [2, 1], [3, 1]],
            columns=["X", "y"],
            index=["a", "b", "c", "d"],
        )
        self.test_data = pd.DataFrame(
            [[0], [1.5], [2.5], [6]], columns=["X"], index=self.train_data.index
        )

        p = ds.Pipeline()
        p.addPipe("read", ds.data.DataPipe())
        p.addPipe("metadata", ds.data.DataPipe(data={"y_true_label": "y"}))
        p.addPipe(
            "clf",
            ds.predictor.SklearnClassifier(clf=KNeighborsClassifier, n_neighbors=3),
            [("read", "data", "df"), ("metadata", "data", "df_metadata")],
        )

        self.pipeline = p

    def test_predict(self):
        p = self.pipeline

        params = {"read": {"data": self.train_data}, "clf": {}}

        p.fit_transform(transform_params=params, fit_params=params)
        self.assertEqual(
            list(p.get_pipe_output("clf")["predict"].index), ["a", "b", "c", "d"]
        )
        self.assertEqual(
            set(p.get_pipe_output("clf")["predict"].columns),
            set(["y", "y_pred_0", "y_pred_1", "y_pred"]),
        )
        self.assertEqual(p.get_pipe_output("clf")["predict"].iloc[0]["y_pred"], 0)
        self.assertEqual(p.get_pipe_output("clf")["predict"].iloc[3]["y_pred"], 1)

        params["read"]["data"] = self.test_data
        p.transform(transform_params=params)
        self.assertEqual(
            list(p.get_pipe_output("clf")["predict"].index), ["a", "b", "c", "d"]
        )
        self.assertEqual(
            set(p.get_pipe_output("clf")["predict"].columns),
            set(["y_pred_0", "y_pred_1", "y_pred"]),
        )
        self.assertEqual(p.get_pipe_output("clf")["predict"].iloc[0]["y_pred"], 0)
        self.assertEqual(p.get_pipe_output("clf")["predict"].iloc[3]["y_pred"], 1)
