from typing import Callable, List, Optional
import abc

import numpy as np

from ..pipe_base import Data, Params, PipeBase
from ..data.csv import MetaData


class FeaturesBase(PipeBase, abc.ABC):
    input_keys = ("df",)
    output_keys = ("df",)

    features: Optional[List[str]] = None

    fit_attributes = [("features", None, None)]


class SpecifyFeaturesBase(FeaturesBase):
    """
    Base class for classes which can be initialised with a list of features or a callable
    which compute those features. The superclass needs to speficify what will be done
    with the feautures during transform.
    """

    features_function: Optional[Callable] = None

    def __init__(
        self, features: List[str] = None, features_function: Callable = None
    ) -> None:
        """
        """
        super().__init__()

        if features_function:
            self.features_function = features_function
        else:
            self.features = features

    def fit_pandas(self, data: Data, params: Params):
        if self.features_function is not None:
            self.features = self.features_function(data["df"])


class DropFeaturesMixin(FeaturesBase):
    """
    Mixin for classes which will drop features. Superclasses needs to
    set self.features, which contains the features which will be dropped.
    """

    def transform_pandas(self, data: Data, params: Params) -> Data:
        del params

        return {"df": data["df"].drop(self.features, axis=1, errors="ignore")}


class DropNonInvertibleFeatures(DropFeaturesMixin, FeaturesBase):
    """
    Drops features that are not invertible, to prevent singularity.
    """

    def fit_pandas(self, data: Data, params: Params):
        self.features = []
        df = data["df"]

        for column in df.columns:
            if not self.is_invertible(df[column]):
                self.features.append(column)

    @staticmethod
    def is_invertible(a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


class DropFeatures(DropFeaturesMixin, SpecifyFeaturesBase):
    pass


class DropHighlyCorrelatedFeatures(DropFeaturesMixin, FeaturesBase):
    """
    When two columns are highly correlated, one will be removed. From
    a pair of correlated columns the one that is the latest one in the list
    of columns, will be removed.
    """

    def __init__(self, threshold: float = 0.9, absolute: bool = True) -> None:
        super().__init__()

        self.threshold = threshold
        self.features = []
        self.absolute = absolute

    def fit_pandas(self, data: Data, params: Params):
        df = data["df"]
        corr_matrix = df.corr()
        if self.absolute:
            corr_matrix = corr_matrix.abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )

        to_drop = [
            column for column in upper.columns if any(upper[column] > self.threshold)
        ]

        self.features = to_drop


class FilterFeatures(SpecifyFeaturesBase):
    """
    FilterFeatures returns a dataframe which contains only the specified
    columns.
    Note: when a request column does not exists in the input dataframe, this
    will be silently ignored.

    """

    input_keys = ("df",)
    output_keys = ("df",)

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        features: List[str] = [] if self.features is None else [
            i for i in self.features if i in df.columns
        ]

        return {"df": df[features]}


class FilterTypeFeatures(PipeBase):
    """
    Keep only the columns of the given type (np.number is default)
    """

    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(self, type_=np.number):
        super().__init__()

        self.type_ = type_

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        features = [
            feature for feature in df.columns if np.issubdtype(df[feature], self.type_)
        ]

        return {"df": df[features]}


class MetadataFilter(FilterFeatures):

    input_keys = ("df", "metadata_df")
    output_keys = ("df",)

    def __init__(self, c: Callable, metadata: MetaData):
        """
        Filter the columns based on metadata

        :param c: a callable which accept a dict with the metadata of a column and return True when the column must be kept
        """
        super().__init__()

        self.c = c
        self.features = [k for k, v in metadata.items() if c(v)]


class ComputeFeature(PipeBase):
    """
    Add a computed feature to the dataframe
    """

    input_keys = ("df",)
    output_keys = ("df",)

    def __init__(self, column_name, f: Callable, c: Callable = None) -> None:
        """
        `f` is a callable whch will get a row of the data and return a
        feature value

        `c` is an optional callable which accepts the df and return True for
        performing this transform and False for skipping
        """
        super().__init__()

        self.column_name = column_name
        self.f = f
        self.c = c

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"].copy()

        if self.c is None or self.c(df):
            df[self.column_name] = df.apply(self.f, axis=1)

        return {"df": df}

    def transform_dask(self, data: Data, params: Params) -> Data:
        df = data["df"]

        if self.c is None or self.c(df):
            df[self.column_name] = df.apply(self.f, axis=1)

        return {"df": df}
