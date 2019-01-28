import seaborn as sns
from IPython.core.display import HTML, display

from ..pipe_base import Data, Params, PipeBase


class SwarmPlots(PipeBase):
    """
    Create swarmplots of all the features in a dataframe.
    This function generates swarmplots on every unique combination of features.
    As the number of features grows, so does the loading time of this function.

    Args:
        data: Dataframe, whose features will be used to create swarm plots.

    Returns:
        Plots all the swarmplots.
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        display(HTML("<h2>Swarmplots Transform %s</h2>" % params["metadata"]["name"]))

        df = data["df"]
        for feature in df.columns:
            for feature2 in df.columns:
                if feature == feature2:
                    continue
                fig = self.get_fig((1, feature, feature2))
                sns.swarmplot(x=feature, y=feature2, data=df)
                # plt.scatter(df[feature], df[feature2], label=params['metadata']['name'])
                # plt.title('Scatterplot of %s and %s' % (feature, feature2))
                # plt.legend()
                # plt.margins(0.02)
                # plt.xlabel(feature)
                # plt.ylabel(feature2)
                display(fig)

        return {"figs": self.figs}

    def transform_dask(self, data, params):
        raise NotImplementedError("Dask dataframe may be too large for swarm plots")
