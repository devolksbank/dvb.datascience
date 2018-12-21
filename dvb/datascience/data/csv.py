import dask.dataframe as dd
import numpy as np
import pandas as pd
from io import StringIO
from typing import Dict
from typing import List

from ..pipe_base import Data, Params, PipeBase


class MetaData:
    varTypes = {
        'ind': np.bool,
        'numi': np.float64,
        'cat': np.str,
        'oms': np.str,
        'cd': np.str,
        'numc': np.float64,
    }

    dtypes: Dict = None

    def __init__(self, file: str):
        self.metadata = pd.read_csv(file, sep=None, engine="python")
        self.dtypes = {}
        for row in self.metadata.itertuples():
            self.dtypes[row.nameVar] = self.varTypes[row.varType]

        self.vars = {}
        for row in self.metadata.itertuples():
            d = {
                'nameVar': row.nameVar,
                'varType': row.varType,
                'PREFIX': row.PREFIX,
                'impMethod': row.impMethod,
                'timing': row.timing,
            }
            self.vars[row.nameVar] = d


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
            file_path: str = None,
            content: str = None,
            sep: bool = None,
            engine: str = "python",
            index_col: str = None,
            metadata: MetaData = None,
            na_values: List = None,

    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.content = content
        self.sep = sep
        self.engine = engine
        self.index_col = index_col
        self.na_values = na_values
        self.metadata = metadata

    @property
    def dtypes(self):
        if self.metadata is None:
            return None

        return self.metadata.dtypes

    def transform_pandas(self, data: Data, params: Params) -> Data:
        content = params.get("content") or self.content
        if content:
            content = StringIO(content)
        file_path = params.get("file_path") or self.file_path
        df = pd.read_csv(
            content or file_path,
            sep=params.get("sep") or self.sep,
            dtype=self.dtypes,
            engine=params.get("engine") or self.engine,
            index_col=params.get("index_col") or self.index_col,
        ).sort_index(axis=1)
        return {"df": df}

    def transform_dask(self, data: Data, params: Params) -> Data:
        content = params.get("content") or self.content
        if content:
            content = StringIO(content)
        file_path = params.get("file_path") or self.file_path
        df = dd.read_csv(
            content or file_path,
            sep=params.get("sep") or self.sep,
            dtype=self.dtypes,
            engine=params.get("engine") or self.engine,
        )
        index_col = params.get("index_col") or self.index_col
        if index_col is not None:
            df.set_index().sort_index(axis=1)

        return {"df": df}


class CSVDataExportPipe(PipeBase):
    """
    Exports a dataframe to CSV.
    Takes as input filepath (str), sep (str).
    Returns a CSV file at the specified location.
    """

    input_keys = ("df",)
    output_keys = ()

    def __init__(self, file_path: str = None, sep: str = None, **kwargs) -> None:
        super().__init__()

        self.file_path = file_path
        self.sep = sep or ","
        self.kwargs = kwargs

    def transform(self, data: Data, params: Params) -> Data:
        data["df"].to_csv(
            params.get("file_path") or self.file_path,
            sep=params.get("sep") or self.sep,
            **self.kwargs
        )
        return {}
