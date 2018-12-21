from typing import List, Optional, Mapping
import abc

import pandas as pd
import numpy as np
from IPython.core.display import display
from sklearn import model_selection
import sklearn.metrics
import ipywidgets
from typing import Tuple

from ..classification_pipe_base import ClassificationPipeBase, Data, Params


class SklearnClassifier(ClassificationPipeBase):
    """
    Wrapper for inclusion of sklearn classifiers in the pipeline.
    """

    input_keys = ("df", "df_metadata")  # type: Tuple[str, ...]
    output_keys = ("predict", "predict_metadata")  # type: Tuple[str, ...]

    threshold = None

    fit_attributes = [("clf", "pickle", "pickle"), ("threshold", None, None)]

    def __init__(self, clf, **kwargs):
        super().__init__()

        self.clf = clf(**kwargs)

    def fit(self, data: Data, params: Params):
        self._set_classification_labels(data["df"], data["df_metadata"])
        X = data["df"][self.X_labels]
        y_true = data["df"][self.y_true_label]

        self.clf.fit(X, y_true)

    def transform(self, data: Data, params: Params) -> Data:
        self._set_classification_data(data["df"], data["df_metadata"])

        self.y_pred_proba = pd.DataFrame(
            self.clf.predict_proba(self.X.values),
            columns=self.y_pred_proba_labels,
            index=self.X.index,
        )

        if params.get("compute_threshold"):
            # compute the threshold in this transform and use it [.. missing words ..] the next transforms
            self.threshold = params["compute_threshold"](
                y_true=self.y_true,
                y_pred_proba=self.y_pred_proba,
                y_pred_proba_labels=self.y_pred_proba_labels,
            )
        predict_metadata = data["df_metadata"].copy()
        predict_metadata["threshold"] = self.threshold

        y_pred = np.zeros(self.X.shape[0])  # type: ignore

        y_pred[
            self.y_pred_proba[self.y_pred_proba_labels[1]] >= (self.threshold or 0.5)
        ] = 1

        self.y_pred_proba[self.y_pred_label] = y_pred

        if self.y_true is not None:
            self.y_pred_proba[self.y_true_label] = self.y_true

        return {"predict": self.y_pred_proba, "predict_metadata": predict_metadata}


class ThresholdBase(abc.ABC):
    """
    What does this do?
    """

    y_true = None  # type: Optional[List]
    y_pred_proba = None  # type: Mapping
    y_pred_proba_labels = None  # type: List[str]

    def set_y(self, y_true, y_pred_proba, y_pred_proba_labels, **kwargs):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred_proba_labels = y_pred_proba_labels

    @abc.abstractmethod
    def __call__(self, **kwargs) -> float:
        pass


class PrecisionRecallThreshold(ThresholdBase):
    def __call__(self, **kwargs) -> float:
        self.set_y(
            kwargs["y_true"], kwargs["y_pred_proba"], kwargs["y_pred_proba_labels"]
        )

        _, _, threshold = sklearn.metrics.precision_recall_curve(
            self.y_true, self.y_pred_proba[self.y_pred_proba_labels[1]]
        )

        return threshold


class CostThreshold(ThresholdBase):
    def __init__(
        self, costFalseNegative: float = 1.0, costFalsePositive: float = 1.0
    ) -> None:
        self.costFalseNegative = costFalseNegative
        self.costFalsePositive = costFalsePositive
        super().__init__()

    def __call__(self, **kwargs) -> float:
        self.set_y(
            kwargs["y_true"], kwargs["y_pred_proba"], kwargs["y_pred_proba_labels"]
        )

        tspace = np.linspace(0, 1, 100)
        costs = []
        for i in range(len(tspace) - 1):
            t = tspace[i]
            costs.append(
                self._computeCost(t, self.costFalseNegative, self.costFalsePositive)
            )
        tmin = tspace[costs.index(min(costs))]
        return tmin

    def _computeConfusionMatrix(self, threshold: float):
        """Make a confusion matrix for the given threshold"""

        y_score = np.zeros(self.y_pred_proba.shape[0])  # type: ignore
        y_score[self.y_pred_proba[self.y_pred_proba_labels[1]] >= threshold] = 1
        return sklearn.metrics.confusion_matrix(self.y_true, y_score)

    def _computeCost(
        self, threshold: float, costFalseNegative: float, costFalsePositive: float
    ) -> float:
        cfmat = self._computeConfusionMatrix(threshold)
        # false positive: payer classified as default
        numFalseNegative = cfmat[1, 0]
        # false negative: default classified paying
        numFalsePositive = cfmat[0, 1]
        cost = (
            numFalseNegative * costFalseNegative + numFalsePositive * costFalsePositive
        )
        return cost


class GridSearchCVProgressBar(model_selection.GridSearchCV):
    """Monkey patch to have a progress bar during grid search"""

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""

        iterator = super()._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)

        cv = model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, "n_splits", 3)
        max_value = n_candidates * n_splits  # count the amount of iterations total

        progress_label = ipywidgets.HTML()
        progress_bar = ipywidgets.FloatProgress(
            min=0, max=max_value, description="GridSearchCV:"
        )
        progress_box = ipywidgets.HBox(
            children=[progress_bar, progress_label]
        )  # setup a progress label + bar

        display(progress_box)
        original_fit = self.estimator.__class__.fit

        def fit(*args, **kwargs):
            progress_bar.value += (
                1
            )  # every time fit is called, increase progress bar by 1
            if (
                progress_bar.value == max_value
            ):  # if max value is reached, display finished and turn green
                progress_label.value = "finished"
                progress_bar.bar_style = "success"

            original_fit(*args, **kwargs)

        self.estimator.__class__.fit = fit

        return iterator


class SklearnGridSearch(SklearnClassifier):

    input_keys = ("df", "df_metadata")
    output_keys = ("predict", "predict_metadata")

    def __init__(self, clf, param_grid, scoring: str = "roc_auc") -> None:
        kwargs = {"estimator": clf(), "param_grid": param_grid, "scoring": scoring}
        self.cv_clf = None
        super().__init__(GridSearchCVProgressBar, **kwargs)

    def fit(self, data: Data, params: Params):
        super().fit(data, params)
        cv_results = pd.DataFrame(self.clf.cv_results_).sort_values(
            by="rank_test_score"
        )
        display("<i>Results of grid search</i>")
        display(cv_results)

        # select the best estimator and store the gridsearch results for optional later inspection
        self.cv_clf = self.clf
        self.clf = self.clf.best_estimator_
        display("<i>Best algoritm</i>")
        display(self.clf)


import tpot


class TPOTClassifier(SklearnClassifier):
    def __init__(self, **kwargs):
        self.clf = tpot.TPOTClassifier(**kwargs)

    def fit(self, data: Data, params: Params):
        super().fit(data, params)

        # select the best estimator and store the gridsearch results for optional later inspection
        self.cv_clf = self.clf
        self.clf = self.clf.fitted_pipeline_
        display("<i>Best algoritm</i>")
        display(self.clf)
