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

    fit_attributes = [("s", "pickle", "pickle")]

    def __init__(self, cls, **kwargs) -> None:
        super().__init__()

        self.cls = cls
        self.s = self.cls(kwargs)
        self.kwargs = kwargs

    def fit_pandas(self, data: Data, params: Params):
        if params.get("kwargs"):
            # kwargs are added, so the cls has to be initialized again
            kwargs = {}
            kwargs.update(self.kwargs)
            kwargs.update(params.get("kwargs", {}))
            self.s = self.cls(kwargs)

        self.s.fit(data["df"])

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()
        r = self.s.transform(df)
        return {"df": pd.DataFrame(r, columns=df.columns, index=df.index)}
