from typing import List

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from ..classification_pipe_base import ClassificationPipeBase, Data, Params

stats.chisqprob = stats.chi2.sf


class GetCoreFeatures(ClassificationPipeBase):
    """
    Get the features (maximum `n_features`) which are key to the label.
    """

    input_keys = ("df", "df_metadata")
    output_keys = ("features",)

    fit_attributes = [("core_features", None, None)]

    def __init__(self, model=None, n_features: int = 10, method="RFE") -> None:
        """
        """
        super().__init__()

        if model is None:
            model = LogisticRegression()

        self.model = model
        self.method = method
        self.n_features = n_features
        self.core_features = []  # type: List[str]

    def fit(self, data: Data, params: Params):
        self._set_classification_labels(data["df"], data["df_metadata"])
        X = data["df"][self.X_labels]
        y = data["df"][self.y_true_label]
        self.core_features = self.get_core_features(X, y)

    def transform(self, data: Data, params: Params) -> Data:
        if self.core_features is None:
            raise ValueError("The fit method has not run.")

        features = self.core_features

        return {"features": features}

    def get_core_features(self, X, y) -> List[str]:
        if self.method == "SFS":
            mySFS = SFS(
                LogisticRegression(),
                k_features=10,
                forward=True,
                cv=0,
                scoring="roc_auc",
            )
            myVars = mySFS.fit(X.values, y.values)
            return [X.columns[i] for i in myVars.k_feature_idx_]

        if self.method == "RFE":
            rfe = RFE(self.model, self.n_features)
            fit = rfe.fit(X, y)
            return [i[1] for i in zip(fit.support_, X.columns) if i[0]]

        raise ValueError("Unknown method for core feature selection")
