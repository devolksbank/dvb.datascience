import statsmodels.api as sm
from IPython.core.display import HTML, display

from ..classification_pipe_base import ClassificationPipeBase, Data, Params


class LogitSummary(ClassificationPipeBase):
    """
    Run a statsmodels logit for coefficient interpretation on the training set

    Args:
        use_regularized (Boolean): Will determine if data is fitted regularized or not (default = True)
        data: Dataframe to be used for this function

    Returns:
        A summary of the logit.
    """

    input_keys = ("df", "df_metadata")
    output_keys = ("summary",)

    def __init__(self, use_regularized: bool = True, **kwargs) -> None:
        super().__init__()

        self.use_regularized = use_regularized
        self.kwargs = kwargs

    def transform(self, data: Data, params: Params) -> Data:
        self._set_classification_labels(data["df"], data["df_metadata"])
        df = data["df"].copy()
        if self.y_true_label not in df:
            return {"summary": None}

        X = df[self.X_labels].copy()
        X["constant"] = 1
        y_true = df[self.y_true_label].copy()

        X = sm.add_constant(X)

        if self.use_regularized:
            summary = sm.Logit(y_true, X).fit_regularized(**self.kwargs).summary()
        else:
            summary = sm.Logit(y_true, X).fit(**self.kwargs).summary()
        display(HTML(summary.as_html()))

        return {"summary": summary}
