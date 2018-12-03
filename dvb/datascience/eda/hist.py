import matplotlib.pyplot as plt
from IPython.core.display import HTML, display

from ..pipe_base import Data, Params, PipeBase


class Hist(PipeBase):
    """
    Create histograms of every feature.

    Args:
        data: Dataframe to be used in creating the histograms
        show_count_labels (Boolean): determines of the number is displayed above every bin (default = True)
        title (str): what title to display above every histogram (default = "Histogram")
        groupBy (str): this string will enable multiple bars in every bin, based on the groupBy column (default = None)
        hit_ratio(bool): show the hit ratio per class based on 1 value of the column `label_column`
        label_column(str): the column to compute the hit_ratio

    Returns:
    Plots of all the histograms.
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def __init__(
        self,
        show_count_labels=True,
        title="Histogram",
        groupBy: str = None,
        hit_ratio: bool = False,
        label_column: str = None,
    ) -> None:
        """
        groupBy: the name of the column to use to make different groups
        """
        self.show_count_labels = show_count_labels
        self.title = title
        self.group_by = groupBy
        self.hit_ratio = hit_ratio
        self.label_column = label_column

        super().__init__()

    def transform(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        display(
            HTML("<h4>%s Transform %s</h4>" % (self.title, params["metadata"]["name"]))
        )

        unique_group_by_values = None
        if self.group_by:
            unique_group_by_values = df[self.group_by].unique()

        for feature in df.columns:
            if feature == self.label_column:
                continue

            if self.group_by is None or feature == self.group_by:
                data = [df[feature]]
                label = [feature]
            else:
                data = tuple(
                    [
                        df[df[self.group_by] == l][feature]
                        for l in unique_group_by_values
                    ]
                )
                label = list(unique_group_by_values)

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
