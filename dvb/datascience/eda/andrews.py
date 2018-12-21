import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
from pandas.plotting import andrews_curves

from ..pipe_base import Data, Params, PipeBase


class AndrewsPlot(PipeBase):
    """
    Create an andrews curves plot of the data in the dataframe

    Args:
        data: Dataframe with the used data
        column: Target column to be used in the andrews curves plot

    Returns:
        The plot.
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def __init__(self, column: str):
        super().__init__()

        self.column = column

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]

        display(
            HTML("<h2>Andrews curves Transform %s</h2>" % params["metadata"]["name"])
        )

        self.get_fig((params["metadata"]["name"]))
        andrews_curves(df, self.column)
        plt.title("Andrews curves")

        return {"figs": self.figs}
