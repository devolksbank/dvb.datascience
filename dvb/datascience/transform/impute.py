from typing import Optional, Any

import numpy as np
import pandas as pd

from ..pipe_base import Data, Params, PipeBase


class ImputeWithDummy(PipeBase):
    """
    Impute missing values with the mean, median, mode or set to a value.
    Takes as input strategy (str). Possible strategies are "mean", "median", "mode" and "value".
    If the strategy is "value", an extra argument impValueTrain can be given, denoting which value should be set.
    """

    input_keys = ("df",)
    output_keys = ("df",)

    possible_strategies = ["mean", "median", "mode", "value"]

    impValueTrain: Optional[Any] = None

    fit_attributes = [("impValueTrain", "pickle", "pickle")]

    def __init__(self, strategy: str = "median", impValueTrain=None) -> None:
        super().__init__()

        if strategy not in self.possible_strategies:
            raise ValueError(
                "Error: strategy {0} not in {1}".format(
                    strategy, self.possible_strategies
                )
            )

        self.strategy = strategy
        self.impValueTrain = impValueTrain

    def fit_pandas(self, data: Data, params: Params):
        df = data["df"]
        if self.strategy == "mean":
            self._set_impute_value_train(df.mean())

        if self.strategy == "median":
            self._set_impute_value_train(df.median())

        if self.strategy == "mode":
            self._set_impute_value_train(df.mode().iloc[0])

    def fit_dask(self, data: Data, params: Params):
        df = data["df"]
        if self.strategy == "mean":
            self._set_impute_value_train(df.mean())

        if self.strategy == "median":
            print("Not Implemented in Dask. Mean is used as fallback")
            self._set_impute_value_train(df.mean())
            # self._set_impute_value_train(df.median())

        if self.strategy == "mode":
            print("Not Implemented in Dask. Mean is used as fallback")
            self._set_impute_value_train(df.mean())
            # self._set_impute_value_train(df.mode().iloc[0])

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]
        return {"df": df.fillna(self.impValueTrain)}

    def _set_impute_value_train(self, impValueTrain):
        if self.impValueTrain is None:
            self.impValueTrain = impValueTrain


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

    def __init__(self, missing_values="NaN", strategy="mode", replacement=""):
        super().__init__()

        self.missing_values = missing_values
        self.replacement = replacement
        self.strategy = strategy
        self.fill = {}

        strategies = ["fixed_value", "mode", "mean"]
        if self.strategy not in strategies:
            raise ValueError(
                "Strategy {0} not in {1}".format(self.strategy, strategies)
            )

        if self.strategy == "fixed_value" and self.replacement is None:
            raise ValueError(
                "Please specify a value for 'replacement'"
                "when using the fixed_value strategy."
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
        for column in data["df"].columns:
            X = data["df"][column]
            mask = self._get_mask(X, self.missing_values)
            X = X[mask.__invert__()]  # unary ~ gives a pylint error
            if self.strategy == "mode":
                mode = pd.Series(X).mode()
                if mode.shape[0] == 0:
                    raise ValueError(
                        "No value is repeated more than once in the column"
                    )
                replacement = mode[0]
            elif self.strategy == "mean":
                replacement = pd.Series(X).mean()
            elif self.strategy == "fixed_value":
                replacement = self.replacement

            self.fill[column] = replacement

    def fit_dask(self, data: Data, params: Params):
        """
        Get the most frequent value.
        """
        for column in data["df"].columns:
            X = data["df"][column]
            mask = self._get_mask(X, self.missing_values)
            X = X[mask.__invert__()]  # unary ~ gives a pylint error
            if self.strategy == "mode":
                replacement = X.value_counts().compute().index[0]
            elif self.strategy == "mean":
                replacement = X.mean()
            elif self.strategy == "fixed_value":
                replacement = self.replacement

            self.fill[column] = replacement

    def transform_pandas(self, data: Data, params: Params) -> Data:
        """
        Replaces missing values in the input data with the most frequent value
        of the training data.
        """
        df = data["df"].copy()

        for column in df.columns:
            mask = self._get_mask(df[column], self.missing_values)
            df[column][mask] = self.fill[column]

        return {"df": df}

    def transform_dask(self, data: Data, params: Params) -> Data:
        """
        Replaces missing values in the input data with the most frequent value
        of the training data.
        """
        df = data["df"]

        for column in df.columns:
            mask = self._get_mask(df[column], self.missing_values)
            df[column] = df[column].mask(mask, self.fill[column])

        return {"df": df}
