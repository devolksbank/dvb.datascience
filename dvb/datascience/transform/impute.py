import warnings
from typing import Optional, Any, List

import numpy as np
import pandas as pd
import dask.dataframe as dd

from ..pipe_base import Data, Params, PipeBase


class ImputeWithDummy(PipeBase):
    """
    Impute missing values with the mean, median, mode or set to a value.
    Takes as input strategy (str). Possible strategies are "mean", "median", "mode" and "value".
    If the strategy is "value", an extra argument impValueTrain can be given, denoting which value should be set.
    Features can contain a list of feature names omn which the action will take place. At default all features are used.
    """

    input_keys = ("df",)
    output_keys = ("df",)

    possible_strategies = ["mean", "median", "mode", "value"]

    impValueTrain: Optional[Any] = None

    fit_attributes = [("impValueTrain", "pickle", "pickle")]

    def __init__(
        self, features: List[str], strategy: str = "median", value=None
    ) -> None:
        super().__init__()

        if strategy not in self.possible_strategies:
            raise ValueError(
                "Error: strategy {0} not in {1}".format(
                    strategy, self.possible_strategies
                )
            )

        if strategy == "median":
            warnings.warn("Median is not Implemented in Dask. Mean is used as fallback")
            strategy = "mean"

        if strategy == "mode":
            warnings.warn("Mode is not Implemented in Dask. Mean is used as fallback")
            strategy = "mean"

        self.strategy = strategy
        self.features = features
        if isinstance(value, dict):
            self.impValueTrain = value
        else:
            self.impValueTrain = {}
            for feature in self.features:
                self.impValueTrain[feature] = value

    def fit_pandas(self, data: Data, params: Params):
        df = data["df"]
        for feature in self.features:
            if self.strategy == "mean":
                self.impValueTrain[feature] = df[feature].mean()

            if self.strategy == "median":
                self.impValueTrain[feature] = df[feature].median()

            if self.strategy == "mode":
                serie = df[feature]
                if isinstance(serie, dd.Series, dd.DataFrame):
                    mode = serie.value_counts().compute().index[0]
                else:
                    mode = serie.mode().iloc[0]

                self.impValueTrain[feature] = mode

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]
        for feature in self.features:
            df[feature] = df[feature].fillna(self.impValueTrain[feature])

        return {"df": df}

    # def transform_dask(self, data: Data, params: Params):
    #     df = self.transform_pandas(data, params)["df"]
    #     if isinstance(df, (dd.Series, dd.DataFrame)):
    #         df = df.compute()
    #     return {"df": df}


class CategoricalImpute(PipeBase):
    """
    Impute missing values from a categorical/string np.ndarray or pd.Series
    with the most frequent value on the training data.

    Args:
        missing_values : string or "NaN", optional (default="NaN")
            The placeholder for the missing values. All occurrences of
            `missing_values` will be imputed. None and np.nan are treated
            as being the same, use the string value "NaN" for them.

        strategy : string, optional (default = 'mode')
            If set to 'mode', replace all instances of `missing_values`
            with the modal value. Otherwise, replace with
            the value specified via `replacement`.

        replacement : string, optional (default='?')
            The value that all instances of `missing_values` are replaced
            with if `strategy` is not set to 'mode'. This is useful if
            you don't want to impute with the mode, or if there are multiple
            modes in your data and you want to choose a particular one. If
            `strategy` is set to `mode`, this parameter is ignored.

    Attributes
    ----------
    fill : str
        Most frequent value of the training data.
    """

    input_keys = ("df",)
    output_keys = ("df",)

    fit_attributes = [("fill", "pickle", "pickle")]

    def __init__(
        self,
        missing_values=None,
        strategy="mode",
        replacement="",
        features: List[str] = None,
    ):
        super().__init__()

        self.missing_values = missing_values if missing_values is not None else ["NaN"]
        self.replacement = replacement
        self.strategy = strategy
        self.features = features
        self.fill = {}

        strategies = ["value", "mode", "mean"]
        if self.strategy not in strategies:
            raise ValueError(
                "Strategy {0} not in {1}".format(self.strategy, strategies)
            )

        if self.strategy == "value" and self.replacement is None:
            raise ValueError(
                "Please specify a value for 'replacement'"
                "when using the value strategy."
            )

    @staticmethod
    def _get_mask(X, value):
        """
        Compute the boolean mask X == missing_values.
        """
        if (
            value == "NaN"
            or value is None
            or (isinstance(value, float) and np.isnan(value))
        ):
            return X.isnull()
        else:
            return X == value

    def fit_pandas(self, data: Data, params: Params):
        """
        Get the most frequent value.
        """
        features = self.features or data["df"].columns
        for feature in features:
            X = data["df"][feature]
            if self.strategy == "mode":
                mode = pd.Series(X).mode(dropna=True)
                if mode.shape[0] == 0:
                    raise ValueError(
                        "No value is repeated more than once in the column"
                    )
                replacement = mode[0]
            elif self.strategy == "mean":
                replacement = pd.Series(X).mean(skipna=True)
            elif self.strategy == "value":
                replacement = self.replacement

            self.fill[feature] = replacement

    @classmethod
    def _compute_df(cls, df):
        if isinstance(df, (dd.DataFrame, dd.Series)):
            return df.compute()

        return df

    def fit_dask(self, data: Data, params: Params):
        """
        Get the most frequent value.
        """
        features = self.features or data["df"].columns
        for feature in features:
            X = data["df"][feature]
            if self.strategy == "mode":
                replacement = self._compute_df(X.value_counts()).index[0]
            elif self.strategy == "mean":
                replacement = X.mean(skipna=True)
            elif self.strategy == "value":
                replacement = self.replacement

            self.fill[feature] = replacement

    def transform_pandas(self, data: Data, params: Params) -> Data:
        """
        Replaces missing values in the input data with the most frequent value
        of the training data.
        """
        df = data["df"].copy()

        features = self.features or data["df"].columns
        for feature in features:
            df[feature] = df[feature].apply(lambda value: self.fill[feature] if value in self.missing_values else value)

        return {"df": df}