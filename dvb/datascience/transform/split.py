from typing import Callable, Any, Optional
import warnings
import logging

from sklearn.model_selection import train_test_split

from ..pipe_base import Data, Params, PipeBase

logger = logging.getLogger(__name__)


class TrainTestSplitBase(PipeBase):
    input_keys = ("df",)
    output_keys = ("df",)

    TRAIN = 0
    TEST = 1
    ALL = -1


class RandomTrainTestSplit(TrainTestSplitBase):
    """
    Return the train, the test set or the complete set, as defined in params['split'].
    The split will be random. A random state is present at default, to make the pipeline reproducable.
    """

    def __init__(self, random_state: int = 42, test_size: float = .25) -> None:
        """
        """
        super().__init__()

        self.random_state = random_state
        self.test_size = test_size

    def transform_pandas(self, data: Data, params: Params) -> Data:
        split = params.get("split", self.ALL)
        if split == self.ALL:
            return {"df": data["df"]}

        test_size = params.get("test_size") or self.test_size
        random_state = params.get("random_state") or self.random_state

        logger.info(
            "TrainTestSplit transforming (test_size=%s, random_state=%s, split=%s)",
            test_size,
            random_state,
            split,
        )

        train, test = train_test_split(
            data["df"], test_size=test_size, random_state=random_state
        )

        return {"df": train if split == self.TRAIN else test}


class TrainTestSplit(RandomTrainTestSplit):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TrainTestSplit is replaced by RandomTrainTestSplit",
            PendingDeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class CallableTrainTestSplit(TrainTestSplitBase):
    """
    Return the train, the test set or the complete set, as defined in params['split'].

    For every row, the callable will be called with the row as single argument and returns
    - CallableTrainTestSplit.TEST
    - CallableTrainTestSplit.TRAIN

    When the return value is not equal to TEST or TRAIN, the row will be excluded from both sets.
    """

    def __init__(self, c: Callable[[Any], int]) -> None:
        """
        """
        super().__init__()

        self.c = c

    def transform_pandas(self, data: Data, params: Params) -> Data:
        split = params.get("split", self.ALL)
        df = data["df"]

        if split == self.ALL:
            return {"df": df}

        train_test_markers = df.apply(self.c, axis=1)
        splitted_df = df[train_test_markers == split]

        logger.info(
            "TrainTestSplit transforming (split=%s, original_size=%s, splitted_size=%s)",
            split,
            len(data["df"]),
            len(splitted_df),
        )

        return {"df": splitted_df}
