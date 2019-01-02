import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import Series as DaskSeries

from ..pipe_base import Data, Params, PipeBase


class CountUniqueValues(PipeBase):
    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(
            self, groupBy: str = None
    ) -> None:
        self.groupBy = groupBy

    def transform_pandas(self, data, params):
        r = {}
        df = data['df']

        for column in df.columns:
            if self.groupBy:
                r[column] = df.groupby([self.groupBy, column]).count()
            else:
                r[column] = df.groupby([column]).count()

        return {'df': r}

    def transform_dask(self, data: Data, params: Params):
        r = {}
        df = data['df']

        for column in df.columns:
            if self.groupBy:
                r[column] = df.groupby([self.groupBy, column]).count().compute()
            else:
                r[column] = df.groupby([column]).count().compute()

        return {'df': r}


class Hist(PipeBase):
    """
    Create histograms of every feature.

    Args:
        data: Dataframe to be used in creating the histograms
        show_count_labels (Boolean): determines of the number is displayed above every bin (default = True)
        title (str): what title to display above every histogram (default = "Histogram")
        groupBy (str): this string will enable multiple bars in every bin, based on the groupBy column (default = None)

    Returns:
    Plots of all the histograms.
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def __init__(
            self, show_count_labels=True, title="Histogram", groupBy: str = None
    ) -> None:
        """
        groupBy: the name of the column to use to make different groups
        """
        self.show_count_labels = show_count_labels
        self.title = title
        self.group_by = groupBy

        super().__init__()

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        display(
            HTML("<h4>%s Transform %s</h4>" % (self.title, params["metadata"]["name"]))
        )

        unique_group_by_values = None
        if self.group_by:
            unique_group_by_values = df[self.group_by].unique()
            if type(unique_group_by_values) is DaskSeries:
                unique_group_by_values = unique_group_by_values.compute()
            unique_group_by_values = [i for i in unique_group_by_values]

        for feature in df.columns:
            if self.group_by is None or feature == self.group_by:
                data = [df[feature]]
                label = [feature]
            else:
                data = []
                for l in unique_group_by_values:
                    data.append(df[df[self.group_by] == l][feature])
                label = list(unique_group_by_values)

            if type(data[0]) in (DaskDataFrame, DaskSeries):
                data = [d.compute() for d in data]

            fig = self.get_fig((1, feature, params["metadata"]["nr"]))
            for idx, d in enumerate(data):
                (values, bins, _) = plt.hist(d, label=[label[idx]], alpha=0.5)
            plt.title(
                "%s (feature %s, transform %s)"
                % (self.title, feature, params["metadata"]["name"])
            )
            plt.legend()
            plt.margins(0.1)
            plt.ylabel("Value")

            if self.show_count_labels:
                for i in range(len(values)):
                    if values[i] > 0:
                        plt.text(
                            bins[i], values[i] + 0.5, str(values[i])
                        )  # 0.5 added for offset values

            display(fig)

        return {"figs": self.figs}

    def transform_dask(self, data, params):
        raise NotImplementedError("Use CountUniqueValues")
