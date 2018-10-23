import pandas as pd
from imblearn.over_sampling import SMOTE

from ..classification_pipe_base import ClassificationPipeBase, Data, Params


class SMOTESampler(ClassificationPipeBase):
    """
    Resample the dataset.

    Note: the new df will not the indexes, because of extra creates row, the indexes
    won't be unique anymore.

    The constructor will pass all kwargs to SMOTE, so all
    arguments of SMOTE can be used. For example:

    ratio (default: 'auto'):
        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    """

    input_keys = ("df", "df_metadata")
    output_keys = ("df",)

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs  # kwargs for SMOTE
        if not "random_state" in self.kwargs:
            self.kwargs["random_state"] = 149

    def fit(self, data: Data, params: Params):
        self._set_classification_labels(data["df"], data["df_metadata"])

    def transform(self, data: Data, params: Params) -> Data:
        self._set_classification_data(data["df"], data["df_metadata"])

        smote = SMOTE(**self.kwargs)

        new_X, new_y_true = smote.fit_sample(self.X, self.y_true)
        new_df = pd.DataFrame(new_X, columns=self.X.columns)
        new_df[self.y_true_label] = new_y_true

        return {"df": new_df}
