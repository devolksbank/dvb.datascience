from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from ..pipe_base import Data, Params, PipeBase


class LabelBinarizerPipe(PipeBase):
    """
    Split label column in different columns per label value
    """

    input_keys = ("df",)
    output_keys = ("df", "df_metadata")

    fit_attributes = [("lb", "pickle", "pickle")]

    lb: Dict[str, LabelBinarizer] = None

    def fit(self, data: Data, params: Params):
        self.lb = {}

        df = data["df"].copy()

        for column in df.columns:
            if df[column].dtype not in (np.str, np.int):
                df[column] = df[column].astype("str")
            self.lb[column] = LabelBinarizer()
            self.lb[column].fit(df[column])

    def transform(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        labels = []
        for column in df.columns:
            df[column] = df[column].to_string()
            labels.append(
                pd.DataFrame(
                    self.lb[column].transform(df[column]),
                    index=df.index,
                    columns=[
                        "%s_%s" % (column, cls) for cls in self.lb[column].classes_
                    ],
                )
            )

        return {
            "df": pd.concat(labels, axis=1, join="outer"),
            "df_metadata": {k: lb.classes_ for k, lb in self.lb.items()},
        }
