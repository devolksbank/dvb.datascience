import pandas as pd
from typing import Callable

from ..pipe_base import PipeBase, Data, Params


class PandasWrapper(PipeBase):
    """
    Generic Wrapper for Pandas operations. The callable will get the DataFrame from the input 'df' and the
    returned DataFrame will be put in the output 'df'.

    Besides the DataFrame, the callable gets the transform_params, so these can be used
    to change the operation.
    """

    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(self, s: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        super().__init__()

        self.s = s

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()
        new_df = self.s(df)
        return {"df": new_df}