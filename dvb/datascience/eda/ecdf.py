import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML, display

from ..pipe_base import Data, Params, PipeBase


class ECDFPlots(PipeBase):
    """
    Creates an empirical cumulative distribution function (ECDF) plot of every feature in the dataframe.

    Args:
        data: Dataframe to be used in the plotting.

    Returns:
        Plots of an ECDF for every feature.
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]

        display(HTML("<h2>ECDF Plots Transform %s</h2>" % params["metadata"]["name"]))

        for feature in df.columns:
            fig = self.get_fig((1, feature))
            plt.plot(
                *self.ecdf(df[feature]),
                marker=".",
                linestyle="none",
                label=params["metadata"]["name"]
            )
            plt.title("ECDF of %s" % feature)
            plt.margins(0.02)
            plt.legend()
            plt.xlabel(feature)
            plt.ylabel("ECDF")
            display(fig)

        return {"figs": self.figs}

    def transform_dask(self, data, params):
        raise NotImplementedError("A Dask dataframe may contain too much data to make an ECDF")

    @staticmethod
    def ecdf(data):
        """
        Compute ECDF for a one-dimensional array of measurements.
        """

        # Number of data points: n
        n = len(data)

        # x-data for the ECDF: x
        x = np.sort(data)

        # y-data for the ECDF: y
        y = np.arange(1, n + 1) / n

        return x, y
