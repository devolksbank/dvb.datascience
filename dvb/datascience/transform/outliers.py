import logging

from typing import List, Dict, Tuple
from pandas.api.types import is_numeric_dtype
from ..pipe_base import PipeBase, Data, Params

logger = logging.getLogger(__name__)


class RemoveOutliers(PipeBase):
    """
    Remove observations when at least one of the features has an outlier.

    Args:
        nr_of_std (int): the number of standard deviations the outlier has to be higher/lower than, to be removed (default = 6)
        skip_columns (List[str]): columns to be skipped
        min_outliers (int): minimum number of outliers a row must have, to be removed from the dataframe (default = 1)

    Returns:
        The dataframe, minus any rows with min_outliers of outliers.
    """

    input_keys = ("df",)
    output_keys = ("df",)

    fit_attributes = [("boundaries", "pickle", "pickle")]

    def __init__(
        self, nr_of_std: int = 6, skip_columns: List[str] = None, min_outliers: int = 1
    ) -> None:
        super().__init__()

        self.nr_of_std = nr_of_std
        self.skip_columns = skip_columns or []
        self.min_outliers = min_outliers

        self.boundaries: Dict[str, Tuple[float, float]] = {}

    def fit_pandas(self, data: Data, params: Params):
        df = data["df"]

        for column in df.columns:
            if column in self.skip_columns:
                continue
            mean = df[column].mean()
            std = df[column].std()
            self.boundaries[column] = (
                mean - self.nr_of_std * std,
                mean + self.nr_of_std * std,
            )

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]

        def keep_observation(row):
            nr_of_found_outliers = 0
            for (column, (lower, upper)) in self.boundaries.items():
                if row[column] < lower:
                    nr_of_found_outliers += 1
                if row[column] > upper:
                    nr_of_found_outliers += 1

                if nr_of_found_outliers >= self.min_outliers:
                    return False

            return True

        return Data({"df": df[df.apply(keep_observation, axis=1)]})


class ReplaceOutliersFeature(PipeBase):
    """
    Replace all outliers in features with the median, mean or a clipped value.

    Args:
        method (str): what method to use when replacing. (default = median)
            Options are:
            - median , replace outliers with median of feature
            - mean , replace outliers with mean of feature
            - clip , replace outliers with nr_of_std standard deviations +/- mean

        nr_of_std (int): minimum number of outliers a row must have, to be removed from the dataframe (default = 1)

    Returns:
        The dataframe, with outliers replaced by the method indicated.
    """

    input_keys = ("df",)
    output_keys = ("df",)

    fit_attributes = [
        ("features_mean", "pickle", "pickle"),
        ("features_median", "pickle", "pickle"),
        ("features_limit", "pickle", "pickle"),
    ]

    supported_methods = ("median", "mean", "clip")

    def __init__(self, method: str = "median", nr_of_std: float = 1.5) -> None:
        """
        """
        super().__init__()

        if not method in self.supported_methods:
            raise ValueError("Method %s is not supported" % method)
        self.method = method
        self.nr_of_std = nr_of_std

        self.features_limit: Dict[str,Tuple] = {}
        self.features_mean: Dict[str,float] = {}
        self.features_median: Dict[str,float] = {}

    def fit_pandas(self, data: Data, params: Params):
        df = data["df"]

        for column in df.columns:
            if not is_numeric_dtype(
                df[column]
            ):  # check if column is only filled with numbers
                continue

            self.features_mean[column] = df[column].mean()

            self.features_median[column] = df[column].median()

            std = df[column].std()
            lower_limit = self.features_mean[column] - (
                std * self.nr_of_std
            )  # lowest value to be considered not an outlier
            if lower_limit < df[column].min():
                lower_limit = df[column].min()

            higher_limit = self.features_mean[column] + (
                std * self.nr_of_std
            )  # highest value to be considered not an outlier
            if higher_limit > df[column].max():
                higher_limit = df[column].max()
            self.features_limit[column] = (lower_limit, higher_limit)

    def fit_dask(self, data: Data, params: Params):
        df = data["df"]

        for column in df.columns:
            if not is_numeric_dtype(
                df[column]
            ):  # check if column is only filled with numbers
                continue
            self.features_mean[column] = df[column].mean().compute()
            std = df[column].std().compute()
            lower_limit = self.features_mean[column] - (
                std * self.nr_of_std
            )  # lowest value to be considered not an outlier
            min_ = df[column].min().compute()
            if lower_limit < min_:
                lower_limit = min_

            higher_limit = self.features_mean[column] + (
                std * self.nr_of_std
            )  # highest value to be considered not an outlier
            max_ = df[column].max().compute()
            if higher_limit > max_:
                higher_limit = max_
            self.features_limit[column] = (lower_limit, higher_limit)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        for column in df.columns:
            df.loc[:, column] = df[column].apply(self._replace_outlier, column=column)

        return Data({"df": df})

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]

        if self.method == "median":
            raise NotImplementedError("Dask does not support median")

        column_dfs = {}
        for column in df.columns:
            column_dfs[column] = df[column].apply(self._replace_outlier, column=column)

        df = df.assign(**column_dfs)

        return Data({"df": df})

    def _replace_outlier(self, x, column):
        try:  # check if x is actually a number
            float(x)
        except (TypeError, ValueError):
            return x

        if x >= self.features_limit[column][0] and x <= self.features_limit[column][1]:
            return x

        if self.method == "mean":
            return self.features_mean[column]

        if self.method == "median":
            return self.features_median[column]

        if (
            self.method == "clip"
        ):  # if value is lower/higher than nearest acceptable change it to nearest acceptable
            if x < self.features_limit[column][0]:
                return self.features_limit[column][0]

            return self.features_limit[column][1]

        raise ValueError("Unknown method %s" % self.method)
