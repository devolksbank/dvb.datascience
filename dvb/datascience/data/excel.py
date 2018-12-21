import pandas as pd
import dask.dataframe as dd

from ..pipe_base import PipeBase, Data, Params


class ExcelDataImportPipe(PipeBase):
    """
    Imports data from excel and creates a dataframe using pd.read_excel().

    Args:
        filepath(str): path to read file
        sheet_name(int): sheet number to be used (default 0)
        index_col(str): index column to be used

    Returns:
        A dataframe with index_col as index column.
    """

    input_keys = ()
    output_keys = ("df",)

    def __init__(
        self, file_path: str = None, sheet_name=0, index_col: str = None
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.sheet_name = sheet_name
        self.index_col = index_col

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = pd.read_excel(
            params.get("file_path") or self.file_path,
            sheet_name=params.get("sheet_name") or self.sheet_name,
            index_col=params.get("index_col") or self.index_col,
        )
        return {"df": df}
