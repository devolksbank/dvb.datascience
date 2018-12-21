import itertools
import logging
from typing import List, Optional, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tabulate
from IPython.core.display import HTML, display
from sklearn.preprocessing import label_binarize

from .classification_pipe_base import ClassificationPipeBase, Data, Params

logger = logging.getLogger(__name__)


class ClassificationScore(ClassificationPipeBase):
    """
    Some scores for classification problems
    """

    input_keys = ("predict", "predict_metadata")
    output_keys = ("scores",)

    # methods which can be used ONLY when y_true is present
    possible_score_methods = [
        "auc",
        "plot_auc",
        "accuracy",
        "mcc",
        "confusion_matrix",
        "plot_confusion_matrix",
        "precision_recall_curve",
        "log_loss",
        "classification_report",
        "plot_model_performance",
    ]

    # methods which can be used when y_true is not present
    possible_predict_methods = ["plot_model_performance"]

    params = None  # type: Params

    def __init__(self, score_methods: List[str] = None) -> None:
        """
        """
        self.score_methods = score_methods or self.possible_score_methods
        super().__init__()

    def fit(self, data: Data, params: Params):
        self._set_classification_labels(data["predict"], data["predict_metadata"])

    def transform(self, data: Data, params: Params) -> Data:
        self._set_classification_data(data["predict"], data["predict_metadata"])

        # assert self.y_true.shape[0] == self.y_pred.shape[0]
        # assert self.y_true.shape[0] == self.y_pred_proba.shape[0]

        self.params = (
            params
        )  # ugly solution to store all params to be accessible by all scores

        score_methods = self.score_methods
        if self.y_true is None:
            score_methods = [
                i for i in score_methods if i in self.possible_predict_methods
            ]

        scores = {}
        for sm in score_methods:
            logger.info("Start scoring method %s", sm)
            scores[sm] = getattr(self, sm)()

        return {"scores": scores}

    def _show_html_table(self, key):
        d = {k: v[key] for k, v in self.transform_data.items()}
        html = tabulate.tabulate([d.values()], d.keys(), tablefmt="html")

        display(HTML("<h2>%s</h2>" % key))
        display(HTML(html))

    def _add_key_value(self, key, value):
        self.transform_data[self.params["metadata"]["name"]][key] = value

    def auc(self) -> Optional[float]:
        if self.n_classes != 2:
            display("auc() not yet implemented for multiclass classifiers")
            return None

        y_score = self.y_pred_proba[self.y_pred_proba_labels[1]]
        r = sklearn.metrics.roc_auc_score(self.y_true, y_score)

        self._add_key_value("auc", r)
        self._show_html_table("auc")

        return r

    def plot_auc(self):
        if self.n_classes != 2:
            display("plot_auc() not yet implemented for multiclass classifiers")
            return None

        # Move binarized to classifier
        y_true_binarized = label_binarize(self.y_true, classes=self.classes)
        y_pred_binarized = 1 - self.y_pred_proba

        y_true_binarized = np.hstack((y_true_binarized, 1 - y_true_binarized))
        y_pred_binarized = np.hstack((y_pred_binarized, 1 - y_pred_binarized))

        fig = plt.figure()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(
                y_true_binarized[:, i], y_pred_binarized[:, i]
            )
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

            # return roc_auc
            self._plot_auc_label(fig, fpr[i], tpr[i], roc_auc[i], i)

        display(HTML("<h2>AUC Plot</h2>"))
        display(fig)

    def _plot_auc_label(self, fig, fpr, tpr, roc_auc, label):
        plt.figure(fig.number)
        lw = 2
        plt.plot(
            fpr, tpr, lw=lw, label="ROC curve (area = %0.2f) %s " % (roc_auc, label)
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            "Receiver operating characteristic (Transform %s)"
            % self.params["metadata"]["name"]
        )
        plt.legend(loc="lower right")
        return fig

    def accuracy(self) -> float:
        r = sklearn.metrics.accuracy_score(self.y_true, self.y_pred)

        self._add_key_value("accuracy", r)
        self._show_html_table("accuracy")

        return r

    def classification_report(self):
        display(HTML("<h2>Classification Report</h2>"))

        clf_rep = sklearn.metrics.precision_recall_fscore_support(
            self.y_true, self.y_pred
        )
        out_dict = {
            "precision": clf_rep[0],
            "recall": clf_rep[1],
            "f1-score": clf_rep[2],
            "support": clf_rep[3],
        }
        out_df = pd.DataFrame(out_dict, index=self.classes)
        avg_tot = (
            out_df.apply(
                lambda x: round(x.mean(), 2)
                if x.name != "support"
                else round(x.sum(), 2)
            )
            .to_frame()
            .T
        )
        avg_tot.index = ["avg/total"]
        out_df = out_df.append(avg_tot)

        display(HTML(out_df.to_html()))

        return out_df

    def confusion_matrix(self):
        return sklearn.metrics.confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_matrix(self):
        fig = plt.figure()
        cm = self.confusion_matrix()
        fmt = "d"

        if self.params.get("normalize", False):
            fmt = ".2f"
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.get_cmap("Blues"))
        plt.title("Confusion Matrix (Transform %s)" % self.params["metadata"]["name"])
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        display(HTML("<h2>Confusion Matrix</h2>"))
        display(fig)

        return fig

    def _average_precision(self):
        return sklearn.metrics.average_precision_score(self.y_true, self.y_pred)

    def _recall(self):
        return sklearn.metrics.recall_score(self.y_true, self.y_pred)

    def _precision(self):
        return sklearn.metrics.precision_score(self.y_true, self.y_pred)

    def precision_recall_curve(self):
        if self.n_classes != 2:
            display(
                "Precision-recall-curve not yet implemented for multiclass classifiers"
            )
            return None

        precision_curve, recall_curve, threshold = sklearn.metrics.precision_recall_curve(
            self.y_true, self.y_pred_proba[self.y_pred_proba_labels[1]]
        )
        precision = self._precision()
        average_precision = self._average_precision()
        recall = self._recall()

        fig = plt.figure()
        from matplotlib.colors import rgb2hex

        color_dark = rgb2hex(plt.cm.get_cmap("Blues")(255))
        color_light = rgb2hex(plt.cm.get_cmap("Blues")(50))
        plt.step(recall_curve, precision_curve, where="post", color=color_dark)
        plt.fill_between(recall_curve, precision_curve, step="post", color=color_light)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.axhline(precision, label="%.2f" % precision, color="r")
        plt.axvline(recall, label="%.2f" % recall, color="b")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend()
        plt.title(
            "2-class Precision-Recall curve: AP={0:0.2f}".format(average_precision)
        )

        display(HTML("<h2>Precision Recall Curve</h2>"))
        display(fig)

        return {
            "precision": precision,
            "recall": recall,
            "threshold": threshold,
            "fig": fig,
        }

    def log_loss(self):
        if self.n_classes != 2:
            display("log_loss() not yet implemented for multiclass classifiers")
            return None

        r = sklearn.metrics.log_loss(self.y_true, self.y_pred)

        self._add_key_value("log_loss", r)
        self._show_html_table("log_loss")

        return r

    def mcc(self, threshold: float = None) -> float:
        if threshold is None:
            y_score = self.y_pred
        else:
            y_score = np.zeros(len(self.y_true))  # type: ignore
            y_score[self.y_pred_proba[self.y_pred_proba_labels[1]] >= threshold] = 1

        r = sklearn.metrics.matthews_corrcoef(self.y_true, y_score)

        self._add_key_value("mcc", r)
        self._show_html_table("mcc")

        return r

    def plot_model_performance(self):
        nr_of_bins = 20
        bins = np.linspace(0, 1, nr_of_bins)

        fig = self.get_fig("model_performance")
        plt.hist(
            self.y_pred_proba[self.y_pred_proba_labels[1]],
            label="predict %s" % self.params["metadata"]["name"],
            bins=bins,
        )

        # if self.y_true is not None:
        #     scores = self.y_pred_proba[self.y_true == 1]
        #     plt.hist(
        #         scores, label="real %s" % self.params["metadata"]["name"], bins=bins
        #     )

        if self.params["metadata"]["nr"] == 1:
            plt.axvline(
                self.threshold,
                color="k",
                linestyle="dashed",
                linewidth=1,
                label="threshold %.2f" % self.threshold,
            )
            plt.title("predictions")
            plt.ylabel("frequency")
            plt.xlabel("model score")

        plt.legend()

        display(HTML("<h2>Model Performance</h2>"))
        display(fig)
