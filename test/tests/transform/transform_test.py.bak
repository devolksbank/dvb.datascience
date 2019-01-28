import unittest

import pandas as pd
from sklearn.preprocessing import StandardScaler

import dvb.datascience as ds
from dvb.datascience.data import csv


class TestTransform(unittest.TestCase):
    def setUp(self):
        # initialize a pipe without the last step
        self.train_csv = "test/data/train.csv"
        self.test_csv = "test/data/test.csv"

        p = ds.Pipeline()
        p.addPipe("read", csv.CSVDataImportPipe())
        p.addPipe(
            "filter_numeric", ds.transform.FilterTypeFeatures(), [("read", "df", "df")]
        )

        self.pipeline = p

    def test_union(self):
        df1 = pd.DataFrame.from_dict(
            data={"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
        )
        df2 = pd.DataFrame.from_dict(
            data={"col_1": [4, 5, 6, 7], "col_2": ["e", "f", "g", "h"]}
        )
        df3 = pd.DataFrame.from_dict(
            data={"col_1": [3, 2, 1, 0], "col_3": ["a", "b", "c", "d"]}
        )

        p = ds.Pipeline()
        p.addPipe("df1", ds.data.DataPipe(data=df1, key="df"))
        p.addPipe("df2", ds.data.DataPipe(data=df2, key="df"))
        p.addPipe("df3", ds.data.DataPipe(data=df3, key="df"))
        p.addPipe(
            "merge_1",
            ds.transform.Union(2, axis=1, remove_duplicated_columns=False),
            [("df1", "df", "df0"), ("df3", "df", "df1")],
        )
        p.addPipe(
            "merge_2",
            ds.transform.Union(2, axis=1, remove_duplicated_columns=True),
            [("df1", "df", "df0"), ("df3", "df", "df1")],
        )
        p.addPipe(
            "merge_3",
            ds.transform.Union(2, axis=0),
            [("df1", "df", "df1"), ("df2", "df", "df1")],
        )
        p.fit_transform()
        self.assertEqual(
            set(p.get_pipe_output("merge_1")["df"].columns),
            set(["col_1", "col_2", "col_3", "col_1_"]),
        )
        self.assertEqual(
            set(p.get_pipe_output("merge_2")["df"].columns),
            set(["col_1", "col_2", "col_3"]),
        )
        self.assertEqual(
            set(p.get_pipe_output("merge_3")["df"].columns), set(["col_1", "col_2"])
        )

    def test_label_binarizer(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "filter", ds.transform.FilterFeatures(["target"]), [("data", "df", "df")]
        )
        p.addPipe(
            "label_binarizer",
            ds.transform.LabelBinarizerPipe(),
            [("filter", "df", "df")],
        )

        p.fit_transform()
        self.assertEqual(
            set(p.get_pipe_output("label_binarizer")["df_metadata"]["target"]),
            set([0, 1, 2]),
        )

        p.transform()
        self.assertEqual(
            set(p.get_pipe_output("label_binarizer")["df"].columns),
            set(["target_0", "target_1", "target_2"]),
        )

    def test_categorical_impute(self):
        data = pd.DataFrame(
            [["yes"], ["no"], [""], ["yes"]], columns=["y"], index=["a", "b", "c", "d"]
        )

        p = ds.Pipeline()
        p.addPipe("data", ds.data.DataPipe("df", data))
        p.addPipe(
            "impute",
            ds.transform.CategoricalImpute(missing_values=""),
            [("data", "df", "df")],
        )

        p.fit_transform()
        self.assertEqual(p.get_pipe_output("impute")["df"]["y"][2], "yes")

        p = ds.Pipeline()
        p.addPipe("data", ds.data.DataPipe("df", data))
        p.addPipe(
            "impute",
            ds.transform.CategoricalImpute(
                missing_values="", strategy="fixed_value", replacement="?"
            ),
            [("data", "df", "df")],
        )

        p.fit_transform()
        self.assertEqual(p.get_pipe_output("impute")["df"]["y"][2], "?")

    def test_random_split(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "split",
            ds.transform.RandomTrainTestSplit(test_size=0.3),
            [("data", "df", "df")],
        )

        p.fit_transform(
            transform_params={
                "split": {"split": ds.transform.RandomTrainTestSplit.TRAIN}
            }
        )
        self.assertEqual(len(p.get_pipe_output("split")["df"]), 105)

        p.transform(
            transform_params={
                "split": {"split": ds.transform.RandomTrainTestSplit.TEST}
            }
        )
        self.assertEqual(len(p.get_pipe_output("split")["df"]), 45)

    def test_callable_split(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "split",
            ds.transform.CallableTrainTestSplit(
                c=lambda row: ds.transform.CallableTrainTestSplit.TRAIN
                if row.name % 2 == 0
                else ds.transform.CallableTrainTestSplit.TEST
            ),
            [("data", "df", "df")],
        )

        p.fit_transform(
            transform_params={
                "split": {"split": ds.transform.RandomTrainTestSplit.TRAIN}
            }
        )
        self.assertEqual(len(p.get_pipe_output("split")["df"]), 75)

        p.transform(
            transform_params={
                "split": {"split": ds.transform.RandomTrainTestSplit.TEST}
            }
        )
        self.assertEqual(len(p.get_pipe_output("split")["df"]), 75)

    def test_remove_outliers(self):
        p = self.pipeline
        p.addPipe(
            "outliers", ds.transform.RemoveOutliers(1), [("filter_numeric", "df", "df")]
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(len(p.get_pipe_output("outliers")["df"]), 2)

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(len(p.get_pipe_output("outliers")["df"]), 0)

    def test_replace_outliers_by_median(self):
        p = self.pipeline
        p.addPipe(
            "outliers",
            ds.transform.ReplaceOutliersFeature("median"),
            [("filter_numeric", "df", "df")],
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[4]["age"], 23)

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[1]["age"], 23)

    def test_replace_outliers_by_mean(self):
        p = self.pipeline
        p.addPipe(
            "outliers",
            ds.transform.ReplaceOutliersFeature("mean"),
            [("filter_numeric", "df", "df")],
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[4]["age"], 29.6)

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[1]["age"], 29.6)

    def test_replace_outliers_by_clip(self):
        p = self.pipeline
        p.addPipe(
            "outliers",
            ds.transform.ReplaceOutliersFeature("clip"),
            [("filter_numeric", "df", "df")],
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(
            p.get_pipe_output("outliers")["df"].iloc[4]["age"], 55.201269499772856
        )

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(
            p.get_pipe_output("outliers")["df"].iloc[1]["age"], 55.201269499772856
        )

    def test_drop_features(self):
        p = self.pipeline
        p.addPipe("drop", ds.transform.DropFeatures("age"), [("read", "df", "df")])

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(
            set(p.get_pipe_output("drop")["df"].columns),
            set(["name", "gender", "length"]),
        )

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(
            set(p.get_pipe_output("drop")["df"].columns),
            set(["name", "gender", "length"]),
        )

    def test_filter_features(self):
        p = self.pipeline
        p.addPipe(
            "filter",
            ds.transform.FilterFeatures(["age", "length"]),
            [("read", "df", "df")],
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(
            set(p.get_pipe_output("filter")["df"].columns), set(["age", "length"])
        )

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(
            set(p.get_pipe_output("filter")["df"].columns), set(["age", "length"])
        )

    def test_compute_feature(self):
        def upper_name(row):
            return row["name"].upper()

        p = self.pipeline
        p.addPipe(
            "compute",
            ds.transform.ComputeFeature("NAME", upper_name),
            [("read", "df", "df")],
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(p.get_pipe_output("compute")["df"].iloc[0]["NAME"], "JAN")

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(p.get_pipe_output("compute")["df"].iloc[0]["NAME"], "GEA")

    def test_core_feature(self):
        p = self.pipeline
        p.addPipe(
            "metadata", ds.data.DataPipe("df_metadata", {"y_true_label": "length"})
        )
        p.addPipe(
            "core",
            ds.transform.GetCoreFeatures(n_features=1),
            [
                ("metadata", "df_metadata", "df_metadata"),
                ("filter_numeric", "df", "df"),
            ],
        )

        p.fit_transform(transform_params={"read": {"file_path": self.train_csv}})
        self.assertEqual(p.get_pipe_output("core")["features"], ["age"])

        p.transform(transform_params={"read": {"file_path": self.test_csv}})
        self.assertEqual(p.get_pipe_output("core")["features"], ["age"])

    def test_sklearn_wrapper(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "split",
            ds.transform.RandomTrainTestSplit(test_size=0.3),
            [("data", "df", "df")],
        )
        p.addPipe(
            "sklearn",
            ds.transform.SKLearnWrapper(StandardScaler),
            [("split", "df", "df")],
        )

        p.fit_transform(
            transform_params={
                "split": {"split": ds.transform.RandomTrainTestSplit.TRAIN}
            }
        )
        self.assertAlmostEqual(
            p.get_pipe_output("sklearn")["df"]["sepal length (cm)"].std(),
            1.00,
            places=2,
        )

        p.transform(
            transform_params={
                "split": {"split": ds.transform.RandomTrainTestSplit.TEST}
            }
        )
        self.assertAlmostEqual(
            p.get_pipe_output("sklearn")["df"]["sepal length (cm)"].std(),
            0.99,
            places=2,
        )

    def test_filter(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "filter",
            ds.transform.FilterObservations(lambda row: row["sepal length (cm)"] > 6),
            [("data", "df", "df")],
        )

        p.fit_transform()
        self.assertEqual(len(p.get_pipe_output("data")["df"]), 150)
        self.assertEqual(len(p.get_pipe_output("filter")["df"]), 61)
