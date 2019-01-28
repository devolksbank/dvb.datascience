from typing import Dict, List, Any
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
import pandas as pd

from ..pipe_base import Data, Params, PipeBase


class CountUniqueValues(PipeBase):
    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(self, groupBy: str = None) -> None:
        super().__init__()

        self.groupBy = groupBy

    def transform_pandas(self, data, params):
        r: Dict[str, int] = {}
        df = data["df"]

        for column in df.columns:
            groups = [column]
            if self.groupBy:
                groups.insert(0, self.groupBy)

            r[column] = df.groupby(groups).count()

        return {"df": r}

    def transform_dask(self, data: Data, params: Params):
        r = {}
        df = data["df"]

        for column in df.columns:
            groups = [column]
            if self.groupBy:
                groups.insert(0, self.groupBy)

            r[column] = df.groupby(groups).count().compute()

        return {"df": r}


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

        unique_group_by_values=  [i for i in df[self.group_by].unique()] if self.group_by else []

        for column in df.columns:
            column_data: List[pd.Series] = []
            if self.group_by is None or column == self.group_by:
                column_data = [df[column]]
                column_labels = [column]
            else:
                column_data = [df[df[self.group_by] == l][column] for l in unique_group_by_values]
                column_labels = list(unique_group_by_values)

            fig = self.get_fig((1, column, params["metadata"]["nr"]))
            for idx, d in enumerate(column_data):
                (values, bins, _) = plt.hist(d, label=[column_labels[idx]], alpha=0.5)
            plt.title(
                "%s (feature %s, transform %s)"
                % (self.title, column, params["metadata"]["name"])
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

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]

        display(
            HTML("<h4>%s Transform %s</h4>" % (self.title, params["metadata"]["name"]))
        )

        unique_group_by_values: List[Any] = []
        if self.group_by:
            unique_group_by_values = [i for i in df[self.group_by].unique().compute()]

        for column in df.columns:
            column_data: List[pd.Series] = []
            if self.group_by is None or column == self.group_by:
                column_data = [df[column].compute()]
                column_labels = [column]
            else:
                column_data = [df[df[self.group_by] == l][column].compute() for l in unique_group_by_values]
                column_labels = list(unique_group_by_values)

            fig = self.get_fig((1, column, params["metadata"]["nr"]))
            for idx, d in enumerate(column_data):
                (values, bins, _) = plt.hist(d, label=[column_labels[idx]], alpha=0.5)
            plt.title(
                "%s (feature %s, transform %s)"
                % (self.title, column, params["metadata"]["name"])
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
