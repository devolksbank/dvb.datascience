import abc
from typing import Any, List, Mapping, Optional, Tuple, Callable, Union

from .pipe_base import Data, Params, PipeBase


class RegressionPipeBase(PipeBase):
    """
    Base class for regression pipes, so regession related attributes and
    methods are reusable for different kind of classification based pipes.
    """

    y_true_label = ""
    y_pred_label = ""
    X_labels: List[str] = None

    fit_attributes: List[Tuple[str, Optional[Union[str, Callable]], Optional[Union[str, Callable]]]] = [
        ("y_true_label", None, None),
        ("y_pred_label", None, None),
        ("X_labels", None, None),
    ]

    @abc.abstractmethod
    def transform(self, data: Data, params: Params) -> Data:
        pass

    def _set_predict_labels(self, df, df_metadata):
        self.y_true_label = df_metadata.get("y_true_label", "y")
        self.y_pred_label = df_metadata.get("y_pred_label", "y_pred")
        self.X_labels = df_metadata.get(
            "X_labels", list(set(df.columns) - set([self.y_true_label]))
        )  # the features, default: all labels except y


    y_true: Optional[List] = None
    y_pred: List = None
    X : Any = None

    def _set_predict_data(self, df, df_metadata):
        self.X = df[self.X_labels]
        self.y_true = None
        if self.y_true_label in df.columns:
            self.y_true = df[self.y_true_label]
        if self.y_pred_label in df.columns:
            self.y_pred = df[self.y_pred_label]
