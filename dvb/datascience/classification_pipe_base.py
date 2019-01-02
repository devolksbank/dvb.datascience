import abc
from typing import Any, List, Mapping, Optional, Tuple, Callable, Union

from .pipe_base import Data, Params, PipeBase


class ClassificationPipeBase(PipeBase):
    """
    Base class for classification pipes, so classification related attributes and
    methods are reusable for different kind of classification based pipes.
    """

    classes: List[str] = None
    n_classes = 0
    y_true_label = ""
    y_pred_label = ""
    y_pred_proba_labels: List[str] = None
    X_labels: List[str] = None

    fit_attributes: List[Tuple[str, Optional[Union[str, Callable]], Optional[Union[str, Callable]]]] = [
        ("classes", None, None),
        ("n_classes", None, None),
        ("y_true_label", None, None),
        ("y_pred_label", None, None),
        ("y_pred_proba_labels", None, None),
        ("X_labels", None, None),
    ]

    @abc.abstractmethod
    def transform(self, data: Data, params: Params) -> Data:
        pass

    def _set_classification_labels(self, df, df_metadata):
        classes = df_metadata.get("classes")

        if classes is None:
            self.classes = [0, 1]
        else:
            self.classes = classes

        self.n_classes = len(self.classes)

        self.y_true_label = df_metadata.get("y_true_label", "y")
        self.y_pred_label = df_metadata.get("y_pred_label", "y_pred")
        self.y_pred_proba_labels = df_metadata.get(
            "y_true_labels", ["y_pred_%s" % c for c in self.classes]
        )

        self.X_labels = df_metadata.get(
            "X_labels", list(set(df.columns) - set([self.y_true_label]))
        )  # the features, default: all labels except y

        self.threshold = df_metadata.get("threshold", 0.5)

    threshold = None
    y_true: Optional[List] = None
    y_pred: List = None
    y_pred_proba: Mapping = None
    X: Any = None

    def _set_classification_data(self, df, df_metadata):
        self.X = df[self.X_labels]
        self.y_true = None
        if self.y_true_label in df.columns:
            self.y_true = df[self.y_true_label]
        if self.y_pred_label in df.columns:
            self.y_pred = df[self.y_pred_label]
            self.y_pred_proba = df[self.y_pred_proba_labels]
