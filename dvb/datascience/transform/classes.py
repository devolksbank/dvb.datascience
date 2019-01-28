from typing import Dict, List

import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import LabelBinarizer

from ..pipe_base import Data, Params, PipeBase


class LabelBinarizerPipe(PipeBase):
    """
    Split label column in different columns per label value
    """

    input_keys = ("df",)
    output_keys = ("df", "df_metadata")

    fit_attributes = [("lb", "pickle", "pickle")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lb : Dict[str, LabelBinarizer] = {}

    def fit_pandas(self, data: Data, params: Params):
        self.lb = {}

        df = data["df"].copy()

        for column in df.columns:
            if df[column].dtype not in (
                np.str,
                np.int,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ):
                df[column] = df[column].astype("str")
            self.lb[column] = LabelBinarizer()
            self.lb[column].fit(df[column])

    def fit_dask(self, data: Data, params: Params):
        self.lb = {}

        df = data["df"]

        for column in df.columns:
            if df[column].dtype not in (
                np.str,
                np.int,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ):
                df[column] = df[column].astype("str")
            self.lb[column] = LabelBinarizer()
            self.lb[column].fit(df[column])

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        binarized_column_dfs: List[pd.DataFrame] = []

        for column in df.columns:
            df[column] = df[column].to_string()
            binarized_column_dfs.append(
                pd.DataFrame(
                    self.lb[column].transform(df[column]),
                    index=df.index,
                    columns=[
                        "%s_%s" % (column, cls) for cls in self.lb[column].classes_
                    ],
                )
            )

        return Data({
            "df": pd.concat(binarized_column_dfs, axis=1, join="outer"),
            "df_metadata": {k: lb.classes_ for k, lb in self.lb.items()},
        })

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]

        binarized_column_arrays: List[np.array] = []
        for column in df.columns:
            df[column] = df[column].to_string()
            binarized_column_arrays.append(self.lb[column].transform(df[column]))

        columns: List[str] = []

        for column in df.columns:
            columns.extend(
                ["%s_%s" % (column, cls) for cls in self.lb[column].classes_]
            )

        df = dd.concat([dd.from_array(l
        # columns=["%s_%s" % (column, cls) for cls in self.lb[column].classes_]
        ) for l in binarized_column_arrays], axis=0, join="outer")
        df.columns = columns

        return Data({
            "df": df,
            "df_metadata": {k: lb.classes_ for k, lb in self.lb.items()},
        })
