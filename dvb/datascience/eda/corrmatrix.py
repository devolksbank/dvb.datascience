import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML, display
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..pipe_base import Data, Params, PipeBase


class CorrMatrixPlot(PipeBase):
    """
    Make a plot of the correlation matrix using all the features in the dataframe

    Args:
        data: Dataframe to be used

    Returns:
        Plot of a correlation matrix
    """

    input_keys = ("df",)
    output_keys = ("fig", "corr")

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]
        labels = df.columns

        display(
            HTML(
                "<h2>Correlation Matrix Transform %s</h2>" % params["metadata"]["name"]
            )
        )

        corr = df.corr()

        fig = self.get_fig((params["metadata"]["name"]))

        fig.set_size_inches(5 + int(len(labels) * 0.3), 5 + int(len(labels) * 0.3))

        axes = plt.gca()

        plt.title("Correlation Matrix")
        plt.margins(0.02)

        myimage = axes.imshow(corr, cmap=plt.cm.get_cmap("bwr"), vmin=-1, vmax=1)

        # create axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(myimage, cmap=plt.cm.get_cmap("bwr"), cax=cax)

        axes.set_xticks(np.arange(0, corr.shape[0], corr.shape[0] / len(labels)))
        axes.set_yticks(np.arange(0, corr.shape[1], corr.shape[1] / len(labels)))

        axes.set_xticklabels(labels, rotation=90)
        axes.set_yticklabels(labels)

        display(fig)

        return {"fig": fig, "corr": corr}
