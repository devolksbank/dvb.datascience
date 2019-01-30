import pathlib
from io import StringIO
from typing import Any, Dict, List, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..pipe_base import Data, Params, PipeBase


class MetaData:
    varTypes = {
        "ind": np.int64,
        "numi": np.float64,
        "cat": np.str,
        "oms": np.str,
        "cd": np.str,
        "numc": np.float64,
        "id": np.str,
    }

    def __init__(self, file: Union[pathlib.Path, str], **kwargs):
        self.metadata = pd.read_csv(file, sep=None, engine="python", **kwargs)
        self.dtypes: Dict[str, Any] = {}
        for row in self.metadata.itertuples():
            self.dtypes[row.varName] = self.varTypes[row.varType]

        self.vars: Dict[str, Dict[str, Any]] = {}
        for row in self.metadata.itertuples():
            d = {
                "varName": row.varName,
                "varType": row.varType,
                "PREFIX": row.PREFIX,
                "impMethod": row.impMethod,
                "impValue": row.impValue,
                "timing": row.timing,
                "dummyForMissings": row.dummyForMissings,
                "auxData": row.auxData,
            }
            self.vars[row.varName] = d

    def items(self):
        return self.metadata.items()


class CSVDataImportPipe(PipeBase):
    """
    Imports data from CSV and creates a dataframe using pd.read_csv().

    Args:
        filepath (str): path to read file
        content (str): raw data to import
        sep (bool): separation character to use
        engine (str): engine to be used, default is "python"
        index_col (str): column to use as index
        na_values: possible values which represent NaN
        metadata: reference to Metadata which is used for dtypes

    Returns:
        A dataframe with index_col as index column.
    """

    input_keys = ()
    output_keys = ("df",)

    metadata = None

    def __init__(
        self,
        file_path: Union[pathlib.Path, str] = None,
        content: str = None,
        sep: bool = None,
        engine: str = "python",
        index_col: str = None,
        metadata: MetaData = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.content = content
        self.sep = sep
        self.engine = engine
        self.index_col = index_col
        self.metadata = metadata
        self.kwargs = kwargs

    @property
    def dtypes(self):
        if self.metadata is None:
            return None

        return self.metadata.dtypes

    def transform_pandas(self, data: Data, params: Params) -> Data:
        content = params.get("content") or self.content
        if content:
            content = StringIO(content)
        file_path: Union[pathlib.Path, str] = params.get("file_path") or self.file_path
        df = pd.read_csv(
            content or file_path,
            sep=params.get("sep") or self.sep,
            dtype=self.dtypes,
            engine=params.get("engine") or self.engine,
            index_col=params.get("index_col") or self.index_col,
            **self.kwargs
        ).sort_index(axis=1)
        return {"df": df}

    def transform_dask(self, data: Data, params: Params) -> Data:
        content = params.get("content") or self.content
        if content:
            content = StringIO(content)
        file_path: Union[pathlib.Path, str] = params.get("file_path") or self.file_path
        df = dd.read_csv(
            content or file_path,
            sep=params.get("sep") or self.sep,
            dtype=self.dtypes,
            engine=params.get("engine") or self.engine,
            **self.kwargs
        )
        index_col = params.get("index_col") or self.index_col
        if index_col is not None:
            df.set_index(self.index_col)

        return {"df": df}


class CSVDataExportPipe(PipeBase):
    """
    Exports a dataframe to CSV.
    Takes as input filepath (str), sep (str).

    In the Dask variant, the fle_path may contain a `*` which will be replaced
    by the filenumber (0, 1, 2, ...) or a name_function.

    >>> CSVDataExportPipe('export-*.csv', name_function=lamda i:  str(date(2015, 1, 1) + i * timedelta(days=1)))

    Returns a CSV file at the specified location.
    """

    input_keys = ("df",)
    output_keys = ()

    def __init__(self, file_path: Union[pathlib.Path, str] = None, sep: str = None, **kwargs) -> None:
        super().__init__()

        self.file_path = file_path
        self.sep = sep or ","
        self.kwargs = kwargs

    def transform_pandas(self, data: Data, params: Params) -> Data:
        data["df"].to_csv(
            params.get("file_path") or self.file_path,
            sep=params.get("sep") or self.sep,
            **self.kwargs
        )
        return {}

    def transform_dask(self, data: Data, params: Params) -> Data:
        if isinstance(data["df"], (pd.Series, pd.DataFrame)):
            return self.transform_pandas(data, params)

        file_path = params.get("file_path") or self.file_path
        if isinstance(file_path, (str, pathlib.Path)) and not "*" in file_path:
            file_path = [file_path]

        data["df"].to_csv(file_path, sep=params.get("sep") or self.sep, **self.kwargs)
        return {}
