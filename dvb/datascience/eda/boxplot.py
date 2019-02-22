from typing import Union, Dict
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
import dask
import numpy as np
import pandas as pd

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

    def _describe_column(
        self, column_data: Union[dask.array.Array, np.ndarray, pd.Series]
    ) -> Dict[int, float]:
        """
        Returns a dict with the main percentiles.
        """
        percentiles = (0, 5, 25, 50, 75, 95, 100)
        d = {}
        if isinstance(column_data, np.ndarray):
            for idx, value in enumerate(np.percentile(column_data, percentiles)):
                d[percentiles[idx]] = value
        elif isinstance(column_data, pd.Series):
            for idx, value in enumerate(
                column_data.describe([p / 100 for p in percentiles])
            ):
                d[percentiles[idx]] = value
        elif isinstance(column_data, pd.Series):
            for idx, value in enumerate(
                dask.array.percentile(column_data, percentiles)
            ):
                d[percentiles[idx]] = value
        else:
            raise ValueError(
                "column_data is of class %s which is not supported"
                % column_data.__class__
            )

        return d

    def _plot_boxplot(
        self,
        percentile_data,
        box_width=0.2,
        whisker_size=20,
        line_width=1.5,
        xoffset=0,
    ):
        """Plots a boxplot from existing percentiles.
        """
        _, ax = plt.subplots()

        x = xoffset

        # box
        y = percentile_data[25]
        height = percentile_data[75] - percentile_data[25]
        box = plt.Rectangle((x - box_width / 2, y), box_width, height)
        ax.add_patch(box)

        # whiskers
        y = (percentile_data[95] + percentile_data[5]) / 2
        v = ax.vlines(x, percentile_data[5], percentile_data[95])

        box.set_linewidth(line_width)
        box.set_facecolor([1, 1, 1, 1])
        box.set_zorder(2)

        v.set_linewidth(line_width)
        v.set_zorder(1)

        whisker_tips = []
        if whisker_size:
            g, = ax.plot(x, percentile_data[5], ls="")
            whisker_tips.append(g)

            g, = ax.plot(x, percentile_data[95], ls="")
            whisker_tips.append(g)

        for wt in whisker_tips:
            wt.set_markeredgewidth(line_width)
            wt.set_markersize(whisker_size)
            wt.set_marker("_")

        g, = ax.plot(x, percentile_data[50], ls="")
        g.set_marker("_")
        g.set_zorder(20)
        g.set_markeredgewidth(line_width)

        return

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]

        display(HTML("<h2>Boxplots Transform %s</h2>" % params["metadata"]["name"]))

        for feature in df.columns:
            fig = self.get_fig((params["metadata"]["name"], feature))

            self._plot_boxplot(self._describe_column(df[feature]))

            plt.title("Boxplot of %s" % feature)
            plt.margins(0.02)
            plt.ylabel("Value")
            display(fig)

        return {"figs": self.figs}