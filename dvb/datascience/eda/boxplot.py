import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

from ..pipe_base import Data, Params, PipeBase


class BoxPlot(PipeBase):
    """
    Create boxplots of every feature in the dataframe.

    Args:
        data: Dataframe to be used in the plotting. Note that only dataframes consisting entirely out of
        integers or floats can be used, as strings cannot be boxplotted.

    Returns:
        Displays the boxplots       .
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]

        display(HTML("<h2>Boxplots Transform %s</h2>" % params["metadata"]["name"]))

        for feature in df.columns:
            fig = self.get_fig((params["metadata"]["name"], feature))
            plt.boxplot([df[feature]])
            plt.title("Boxplot of %s" % feature)
            plt.margins(0.02)
            plt.ylabel("Value")
            display(fig)

        return {"figs": self.figs}

