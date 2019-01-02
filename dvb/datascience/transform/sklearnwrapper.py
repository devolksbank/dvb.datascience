from typing import Any

import pandas as pd

from ..pipe_base import Data, Params, PipeBase


class SKLearnBase:
    def fit(self, data: Any):
        pass

    def transform(self, data: Any):
        pass


class SKLearnWrapper(PipeBase):
    """
    Generic SKLearn fit / transform wrapper
    Geen idee wat dit precies doet...
    """

    input_keys = ("df",)
    output_keys = ("df",)

    s: SKLearnBase = None

    fit_attributes = [("s", "pickle", "pickle")]

    def __init__(self, cls, **kwargs) -> None:
        super().__init__()

        self.cls = cls
        self.kwargs = kwargs

    def fit(self, data: Data, params: Params):
        kwargs = {}
        kwargs.update(self.kwargs)
        kwargs.update(params.get("kwargs", {}))  # for grid search
        self.s = self.cls(kwargs)
        self.s.fit(data["df"])

    def transform(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()
        r = self.s.transform(df)
        return {"df": pd.DataFrame(r, columns=df.columns, index=df.index)}
