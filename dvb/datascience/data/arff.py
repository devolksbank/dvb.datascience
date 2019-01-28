from typing import Union
import pathlib

import numpy as np
import pandas as pd
from scipy.io import arff

from ..pipe_base import PipeBase, Data, Params


class ARFFDataImportPipe(PipeBase):
    """
    Imports ARFF files and returns a dataframe.

    Args:
        file_path (str): String with a path to the file to import

    Returns:
        A dataframe
    """

    input_keys = ()
    output_keys = ("df",)

    file_path: Union[pathlib.Path, str] = None

    fit_attributes = [("file_path", None, None)]

    def fit(self, data: Data, params: Params):
        self.file_path = params["file_path"]

    def transform_pandas(self, data: Data, params: Params) -> Data:
        arff_data = arff.loadarff(self.file_path)[0]

        return {"df": pd.DataFrame(arff_data)}


class ARFFDataExportPipe(PipeBase):
    """
    Exports ARFF files and writes it to file.

    Args:
        file_path (str): String with a path to the file to import
        wekaname (str): The wekaname to be used
    Returns:
        A file.
    """

    input_keys = ("df",)
    output_keys = ()

    file_path: Union[pathlib.Path, str] = None
    wekaname: str = None

    fit_attributes = [("file_path", None, None), ("wekaname", None, None)]

    def fit(self, data: Data, params: Params):
        self.file_path = params["file_path"]
        self.wekaname = params["wekaname"]

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]

        if not isinstance(self.wekaname, str):
            raise ValueError("wekaname is not a string")

        with open(self.file_path, "w") as f:
            arffList = []
            arffList.append("@relation " + self.wekaname + "\n")
            # look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
            for i in range(df.shape[1]):
                if df.dtypes[i] == "O" or (
                        df.columns[i] in ["Class", "CLASS", "class"]
                ):
                    _uniqueNominalVals = [str(_i) for _i in np.unique(df.iloc[:, i])]
                    _uniqueValuesString = "{%s}" % ",".join(_uniqueNominalVals).replace(
                        "[", ""
                    ).replace("]", "")
                    arffList.append(
                        "@attribute " + df.columns[i] + _uniqueValuesString + "\n"
                    )
                else:
                    arffList.append("@attribute " + df.columns[i] + " real\n")
                    # even if it is an integer, let's just deal with it as a real number for now

            arffList.append("@data\n")
            for i in range(df.shape[0]):  # instances
                _instanceString = ""
                for j in range(df.shape[1]):  # features
                    if df.dtypes[j] == "O":
                        _instanceString += '"' + str(df.iloc[i, j]) + '"'
                    else:
                        _instanceString += str(df.iloc[i, j])

                    if (
                            j != df.shape[1] - 1
                    ):  # if it's not the last feature, add a comma
                        _instanceString += ","
                _instanceString += "\n"
                arffList.append(_instanceString)
            f.writelines(arffList)

        return {}
