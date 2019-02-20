from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd
import dask.dataframe as dd

from ..pipe_base import Data, Params, PipeBase


class EncodingBase:
    fit_attributes = [
        ("encoding", "pickle", "pickle"),
        ("encoding", "pickle", "pickle"),
    ]

    def __init__(self, features: List[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.features = features
        self.encoding: Dict[str, Dict[any, int]] = {}
        self.decoding: Dict[str, Dict[int, any]] = {}

    def _make_encode(self, df_f, feature):
        unique = df_f.unique()
        if isinstance(unique, (dd.DataFrame, dd.Series)):
            unique = unique.compute()
        encoding = {value: idx for idx, value in enumerate(unique)}
        decoding = {idx: value for idx, value in enumerate(unique)}
        self.encoding[feature] = encoding
        self.decoding[feature] = decoding

    def fit_pandas(self, data: Data, params: Params):
        df = data["df"].copy()

        for feature in self.features or df.columns:
            self._make_encode(df[feature], feature)

    def fit_dask(self, data: Data, params: Params):
        df = data["df"]

        for feature in self.features or df.columns:
            self._make_encode(df[feature], feature)


class LabelBinarizerPipe(EncodingBase, PipeBase):
    """
    Split label column in different columns per label value
    """

    input_keys = ("df",)
    output_keys = ("df", "df_metadata")

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        df_metadata = defaultdict(list)

        for feature in self.features or df.columns:
            for class_ in self.encoding[feature].values():
                new_feature = "%s_%s" % (feature, class_)
                df[new_feature] = self._series_apply(
                    series=df[feature],
                    f=lambda row: int(row == class_),
                    meta=pd.Series(dtype="int", name=new_feature),
                )
                df_metadata["target"].append(new_feature)

        return {"df": df, "df_metadata": df_metadata}

    def _series_apply(self, series, f, meta, **kwargs):
        if isinstance(series, pd.Series):
            return series.apply(f, **kwargs)

        return series.apply(f, meta=meta, **kwargs)

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]

        df_metadata = defaultdict(list)

        for feature in self.features or df.columns:
            for class_ in self.encoding[feature].values():
                new_feature = "%s_%s" % (feature, class_)
                df[new_feature] = self._series_apply(
                    series=df[feature],
                    f=lambda row: int(row == class_),
                    meta=pd.Series(dtype="int", name=new_feature),
                )
                df_metadata[feature].append(new_feature)

        return {"df": df, "df_metadata": df_metadata}


class LabelEncoderPipe(EncodingBase, PipeBase):
    """
    Replace categorical with number
    """

    input_keys = ("df",)
    output_keys = ("df",)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        for feature in self.features:
            df[feature] = df.apply(
                lambda row: self.encoding[feature][row[feature]],
                axis=1,
                meta=pd.Series(dtype="int", name=feature),
            )

        return {"df": df}

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]

        for feature in self.features:
            df[feature] = df.apply(
                lambda row: self.encoding[feature][row[feature]],
                axis=1,
                meta=pd.Series(dtype="int", name=feature),
            )

        return {"df": df}
