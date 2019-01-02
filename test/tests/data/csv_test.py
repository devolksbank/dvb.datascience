import pytest
import unittest
import pathlib

import pandas as pd
import dask.dataframe as dd

import dvb.datascience as ds


class TestCsv(unittest.TestCase):
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
        ).sort_index(axis=1)

        if self.pipeline.dataframe_engine == "dask":
            self.train_data = dd.from_pandas(self.train_data, chunksize=1).compute()

        self.file_path = "test/data/train.csv"
        with open(self.file_path) as f:
            self.content = f.read()

    def _compute_when_needed(self, df):
        if self.pipeline.dataframe_engine == 'dask':
            return df.compute()

        return df

    def test_read_file(self):
        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe(file_path=self.file_path))
        p.transform()
        df = p.get_pipe_output("read")["df"]
        df = self._compute_when_needed(df)
        # make sure the order of the columns is correct
        df = df[["age", "gender", "length", "name"]]
        self.assertTrue(pd.DataFrame.equals(df, self.train_data))

    @pytest.mark.skipif(
        ds.pipeline.default_dataframe_engine == "dask",
        reason="Dask does not support reading from content",
    )
    def test_read_content(self):
        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe(content=self.content))
        p.transform()
        df = p.get_pipe_output("read")["df"]
        self.assertTrue(pd.DataFrame.equals(df, self.train_data))

    def test_read_init_params(self):
        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe())
        p.transform(
            data=None, transform_params={"read": {"file_path": "test/data/train.csv"}}
        )
        df = p.get_pipe_output("read")["df"]
        df = self._compute_when_needed(df)
        # make sure the order of the columns is correct
        df = df[["age", "gender", "length", "name"]]
        self.assertTrue(pd.DataFrame.equals(df, self.train_data))

    def test_read_custom_separator(self):
        pass

    def test_read_custom_agent(self):
        pass

    def test_read_custom_index(self):
        pass

    def test_write(self):
        pathlib.Path("./tmp").mkdir(parents=True, exist_ok=True)
        output_file = "tmp/unittest-csv_test-test_write_output.csv"

        p = self.pipeline
        p.addPipe("read", ds.data.CSVDataImportPipe(file_path="test/data/test.csv"))
        p.addPipe(
            "write",
            ds.data.CSVDataExportPipe(file_path=output_file),
            [("read", "df", "df")],
        )
        p.transform()
        output = p.get_pipe_output("write")
        self.assertEqual(output, dict())

        # Inspect the file on disk
        with open(output_file, "r") as content_file:
            content = content_file.read().split('\n')
            self.assertEqual(set(content[0].split(',')), set(["", "age", "gender", "length", "name"]))
            self.assertEqual(set(content[1].split(',')), set(["0", "25", "W", "161", "gea"]))

    def test_write_init_params(self):
        pass

    def test_write_custom_separator(self):
        pass
