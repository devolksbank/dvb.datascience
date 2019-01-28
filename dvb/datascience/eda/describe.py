from IPython.core.display import display

from .base import AnalyticsBase
from ..pipe_base import Data, Params


class Describe(AnalyticsBase):
    """
    Describes the data.

    Args:
        data: Dataframe to be used.

    Returns:
        A description of the data.
    """

    input_keys = ("df",)
    output_keys = ("output",)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        output = data["df"].describe()
        display(output)
        return {"output": output}

    def transform_dask(self, data: Data, params: Params) -> Data:
        output = data["df"].describe()
        display(output)
        return {"output": output}
