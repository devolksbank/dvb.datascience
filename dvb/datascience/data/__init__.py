from typing import Dict, Any, Optional
import pandas as pd
import sklearn.datasets
import sklearn.utils

from .arff import ARFFDataExportPipe
from .arff import ARFFDataImportPipe
from .csv import MetaData
from .csv import CSVDataExportPipe
from .csv import CSVDataImportPipe
from .excel import ExcelDataImportPipe
from .teradata import TeraDataImportPipe
from ..pipe_base import PipeBase, Data, Params


class DataPipe(PipeBase):
    """
    Add some data to the pipeline via fit or transform params. The data can be added on three different moments:

    >>> pipe = DataPipe(data=[1,2,3])
    >>> pipeline.fit_transform(fit_params={"data": [4,5,6]})
    >>> pipeline.transform(transform_params={"data": [7,8,9]})

    The last data will be used.
    """

    def __init__(self, key: str = "data", data=None) -> None:
        super().__init__()

        self.key = key
        self.input_keys = ()
        self.output_keys = (key,)
        self.data = data
        self.fitted_data = None  # type: Optional[Data]

    def fit(self, data: Data, params: Params):
        self.fitted_data = self.data
        if "data" in params:
            self.fitted_data = params["data"]

    def transform(self, data: Data, params: Params) -> Data:
        if "data" in params:
            data = params["data"]
        else:
            data = self.fitted_data if self.fitted_data is not None else {}

        if data is None:
            raise ValueError(
                "data need to be given at initialization, fitting or transforming"
            )

        return {self.key: data}


sampleDatasets = {
    "iris": {
        # Classes	3
        # Samples per class	50
        # Samples total	150
        # Dimensionality	4
        # Features	real, positive
        "classes": ["Setosa", "Versicolour", "Virginica"],
        "y_true_label": "target",
    },
    "diabetes": {
        # Samples total	442
        # Dimensionality	10
        # Features	real, -.2 < x < .2
        # Targets	integer 25 - 346
        "classes": None,
        "y_true_label": "target",
    },
    "wine": {
        # Classes	3
        # Samples per class	[59,71,48]
        # Samples total	178
        # Dimensionality	13
        # Features	real, positive
        "classes": None,
        "y_true_label": "target",
    },
    "boston": {
        # Samples total	506
        # Dimensionality	13
        # Features	real, positive
        # Targets	real 5. - 50.
        "classes": None,
        "y_true_label": "target",
    },
    "breast_cancer": {
        # Classes	2
        # Samples per class	212(M),357(B)
        # Samples total	569
        # Dimensionality	30
        # Features	real, positive
        "classes": [0, 1],
        "y_true_label": "target",
    },
    "digits": {
        # Classes	10
        # Samples per class	~180
        # Samples total	1797
        # Dimensionality	64
        # Features	integers 0-16
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "y_true_label": "target",
    },
    "linnerud": {
        # Samples total	20
        # Dimensionality	3 (for both data and target)
        # Features	integer
        # Targets	integer
        "classes": None,
        "y_true_label": "target",
    },
}  # type: Dict[str, Dict[str, Any]]


class SampleData(PipeBase):

    input_keys = ()
    output_keys = ("df", "df_metadata")

    possible_dataset_names = list(sampleDatasets.keys())

    def __init__(self, dataset_name: str = "iris") -> None:
        super().__init__()

        if dataset_name not in self.possible_dataset_names:
            raise ValueError("Unknown dataset %s" % dataset_name)

        self.dataset_name = dataset_name
        self.y_true_label = sampleDatasets[dataset_name]["y_true_label"]
        self.classes = sampleDatasets[dataset_name]["classes"]

    def transform(self, data: Data, params: Params) -> Data:
        d = getattr(
            sklearn.datasets, "load_" + self.dataset_name
        )()  # type: sklearn.utils.Bunch
        df = pd.DataFrame(
            d.data, columns=getattr(d, "feature_names", None)  # pylint: disable=e1101
        )
        if len(d.target.shape) == 1:  # pylint: disable=e1101
            df[self.y_true_label] = pd.Series(d.target)  # pylint: disable=e1101
        else:
            df_target = pd.DataFrame(
                d.target,  # pylint: disable=e1101
                columns=getattr(d, "target_names", None),  # pylint: disable=e1101
            )
            df = df.join(df_target)

        r = {"df": df, "df_metadata": sampleDatasets[self.dataset_name]}

        if hasattr(d, "target_names"):
            r["y_names"] = d.target_names  # pylint: disable=e1101
        if hasattr(d, "images"):
            r["images"] = d.images  # pylint: disable=e1101

        return r


class GeneratedSampleClassification(PipeBase):

    input_keys = ()
    output_keys = ("df", "df_metadata")

    def __init__(
        self,
        n_classes: int = 10,
        n_features: int = 20,
        n_samples: int = 100,
        random_state: int = None,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.random_state = random_state

    def transform(self, data: Data, params: Params) -> Data:
        X, y = sklearn.datasets.make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            random_state=self.random_state,
            n_redundant=0,
            n_clusters_per_class=1,
            n_informative=1,
        )

        df = pd.DataFrame(X, columns=range(self.n_features))
        df["target"] = y

        return {
            "df": df,
            "df_metadata": {"classes": range(self.n_classes), "y_true_label": "target"},
        }
