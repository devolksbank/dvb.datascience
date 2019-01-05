import pytest

import numpy as np
import numpy.testing
from sklearn.neighbors import KNeighborsClassifier

import dvb.datascience as ds


@pytest.mark.usefixtures("dataframe_engine")
class TestScoreMethods:
    def get_pipeline(self, dataset_name, clf_pipe) -> ds.Pipeline:
        p = ds.Pipeline()
        p.addPipe("read", ds.data.SampleData(dataset_name=dataset_name))
        p.addPipe(
            "split",
            ds.transform.RandomTrainTestSplit(test_size=0.3, random_state=42),
            [("read", "df", "df")],
        )
        p.addPipe(
            "clf",
            clf_pipe,
            [("split", "df", "df"), ("read", "df_metadata", "df_metadata")],
        )
        p.addPipe(
            "score",
            ds.score.ClassificationScore(["accuracy", "confusion_matrix"]),
            [
                ("clf", "predict", "predict"),
                ("clf", "predict_metadata", "predict_metadata"),
            ],
        )

        return p

    def test_binaryclass_kneighbors(self):
        p = self.get_pipeline(
            "breast_cancer",
            ds.predictor.SklearnClassifier(clf=KNeighborsClassifier, n_neighbors=3),
        )

        params = {"split": {"split": ds.transform.RandomTrainTestSplit.TRAIN}}

        p.fit_transform(transform_params=params, fit_params=params)
        assert round(abs(p.get_pipe("score").accuracy()-0.947), 3) == 0
        numpy.testing.assert_array_equal(
            p.get_pipe("score").confusion_matrix(), np.array([[133, 16], [5, 244]])
        )
        assert round(abs(p.get_pipe("score").auc()-0.992), 3) == 0
        assert round(abs(p.get_pipe("score").mcc()-0.887), 3) == 0
        assert round(abs(p.get_pipe("score").log_loss()-1.822), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["f1-score"]-0.959), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["precision"]-0.938), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["recall"]-0.980), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["support"]-249.0), 3) == 0

        params = {"split": {"split": ds.transform.RandomTrainTestSplit.TEST}}
        p.transform(transform_params=params)
        assert round(abs(p.get_pipe("score").accuracy()-0.942), 3) == 0
        numpy.testing.assert_array_equal(
            p.get_pipe("score").confusion_matrix(), np.array([[57, 6], [4, 104]])
        )
        assert round(abs(p.get_pipe("score").auc()-0.991), 3) == 0
        assert round(abs(p.get_pipe("score").mcc()-0.874), 3) == 0
        assert round(abs(p.get_pipe("score").log_loss()-2.020), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["f1-score"]-0.954), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["precision"]-0.945), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["recall"]-0.963), 3) == 0
        assert round(abs(p.get_pipe("score").classification_report().iloc[1]["support"]-108.0), 3) == 0