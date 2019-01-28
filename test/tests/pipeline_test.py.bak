import unittest
import json
import tempfile
import shutil
import os.path

import dvb.datascience as ds


class TestPipes(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def testChainPipes(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris")).addPipe(
            "split", ds.transform.RandomTrainTestSplit(test_size=0.3), [("data", "df", "df")]
        )

        self.assertEqual(len(p.pipes), 2)

    def testIsPipeInputForAnotherPipe(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        self.assertFalse(p._is_pipe_input_for_another_pipe(p.get_pipe("data")))
        p.addPipe(
            "split",
            ds.transform.RandomTrainTestSplit(test_size=0.3),
            [("data", "df", "df")],
        )
        self.assertFalse(p._is_pipe_input_for_another_pipe(p.get_pipe("split")))
        self.assertTrue(p._is_pipe_input_for_another_pipe(p.get_pipe("data")))

    def testPipeline(self):
        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "split", ds.transform.RandomTrainTestSplit(test_size=0.3), [("data", "df", "df")]
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

    def testPipelineOutputToFile(self):
        p = ds.Pipeline(output_store=ds.OutputStoreFile())
        p.addPipe("data", ds.data.SampleData("iris"))
        p.addPipe(
            "split", ds.transform.RandomTrainTestSplit(test_size=0.3), [("data", "df", "df")]
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

    def testSaveLoadPipeline(self):
        p = ds.Pipeline()
        p.addPipe("read", ds.data.csv.CSVDataImportPipe())
        p.addPipe(
            "filter_numeric", ds.transform.FilterTypeFeatures(), [("read", "df", "df")]
        )
        p.addPipe(
            "outliers",
            ds.transform.ReplaceOutliersFeature("median"),
            [("filter_numeric", "df", "df")],
        )

        p.fit_transform(transform_params={"read": {"file_path": "test/data/train.csv"}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[4]["age"], 23)

        p.transform(transform_params={"read": {"file_path": "test/data/test.csv"}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[1]["age"], 23)

        file_path = os.path.join(self.test_dir, "export.json")

        p.save(file_path)
        with open(file_path) as f:
            state = json.load(f)
            self.assertEqual(
                ds.PipeBase._string_base64_pickle(state["outliers"]["features_median"]),
                {"age": 23.0, "length": 180.0},
            )

        p.load(file_path)
        p.transform(transform_params={"read": {"file_path": "test/data/test.csv"}})
        self.assertEqual(p.get_pipe_output("outliers")["df"].iloc[1]["age"], 23)

    def testSubPipeline(self):
        features = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        featuresSimplifiedNames = {
            features[0]: "sl",
            features[1]: "sw",
            features[2]: "pl",
            features[3]: "pw",
        }

        p = ds.Pipeline()
        p.addPipe("data", ds.data.SampleData("iris"))

        class TestSubPipline(ds.sub_pipe_base.SubPipelineBase):

            input_keys = ("df",)
            output_keys = ("df",)

            def __init__(self):
                super().__init__("union")

                connectorList = []

                for idx, feature in enumerate(features):
                    self.sub_pipeline.addPipe(
                        "filter_" + featuresSimplifiedNames[feature],
                        ds.transform.FilterFeatures([feature]),
                        [("pass_data", "df", "df")],
                    )
                    self.sub_pipeline.addPipe(
                        "outlier_" + featuresSimplifiedNames[feature],
                        ds.transform.ReplaceOutliersFeature(),
                        [("filter_" + featuresSimplifiedNames[feature], "df", "df")],
                    )
                    connectorList.append(
                        (
                            "outlier_" + featuresSimplifiedNames[feature],
                            "df",
                            "df%s" % idx,
                        )
                    )
                print(connectorList)
                self.sub_pipeline.addPipe("union", ds.transform.Union(4), connectorList)

        p.addPipe("sub", TestSubPipline(), [("data", "df", "df")])

        p.fit_transform()
        self.assertEqual(
            len(
                p.get_pipe("sub").sub_pipeline.get_pipe_output(
                    "filter_" + featuresSimplifiedNames[features[0]]
                )["df"]
            ),
            150,
        )
        self.assertEqual(len(p.get_pipe_output("sub")["df"]), 150)

        p.transform()
        self.assertEqual(
            len(
                p.get_pipe("sub").sub_pipeline.get_pipe_output(
                    "filter_" + featuresSimplifiedNames[features[0]]
                )["df"]
            ),
            150,
        )
        self.assertEqual(len(p.get_pipe_output("sub")["df"]), 150)
