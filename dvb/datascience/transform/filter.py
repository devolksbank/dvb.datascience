from typing import Callable

from ..pipe_base import Data, Params, PipeBase


class FilterObservations(PipeBase):
    """
    Filter observations by row based on a function
    """

    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(self, filter_: Callable) -> None:
        """
        `filter` is a callable function which gets a row and return True when that row needs to be kept.
        """
        super().__init__()

        self.filter_ = filter_

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()
        return {"df": df[df.apply(self.filter_, axis=1)]}

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]
        return {"df": df[df.apply(self.filter_, axis=1)]}
