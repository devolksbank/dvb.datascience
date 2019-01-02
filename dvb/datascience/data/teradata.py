import decimal
import logging
import time

import pandas as pd
import dask.dataframe as dd

from ..pipe_base import PipeBase, Data, Params

try:
    import teradata
except ImportError:
    use_teradata = False
else:
    use_teradata = True

logger = logging.getLogger(__name__)

if not use_teradata:
    TeraDataImportPipe = None
else:

    class customDataTypeConverter(teradata.datatypes.DefaultDataTypeConverter):
        """
        Transforms data types from Teradata to datatypes used by Python.
        Replaces decimal comma with decimal point.
        Changes BYTEINT, BIGINT, SMALLINT and INTEGER to the Python type int.
        """

        def __init__(self):
            super().__init__(useFloat=True)

        def convertValue(self, dbType, dataType, typeCode, value):
            if value is not None and dataType == "DECIMAL":
                return decimal.Decimal(value.replace(",", "."))

            if value is not None and (
                dataType == "BYTEINT"
                or dataType == "BIGINT"
                or dataType == "SMALLINT"
                or dataType == "INTEGER"
            ):
                return int(value)

            return super().convertValue(dbType, dataType, typeCode, value)

    class TeraDataImportPipe(PipeBase):
        """
        Reads data from Teradata and returns a dataframe.

        Args:
            file_path(str): path to read file containing SQL query
            sql(str): raw SQL query to be used

        Returns:
            A dataframe using pd.read_sql_query(), sorts the index alphabetically.
        """

        def __init__(self):
            super().__init__()
            if not use_teradata:
                logger.error("Teradata module is not imported and could not be used")

        def _transform(self, params: Params, read_sql_query) -> Data:
            sql = params.get("sql", "")
            if params["file_path"]:
                with open(params["file_path"], "r") as f:
                    sql = f.read()

            start = time.time()
            udaExec = teradata.UdaExec(
                appName="td",
                version="1.0",
                configureLogging=False,
                logConsole=False,
                logLevel="TRACE",
                dataTypeConverter=customDataTypeConverter(),
            )
            conn = udaExec.connect(method="odbc", DSN="Teradata")
            df = read_sql_query(sql, conn)
            logger.info(
                "teradata returned %s rows in % seconds",
                str(len(df)),
                str(round(time.time() - start)),
            )
            conn.close()
            return {"df": df}

        def transform_pandas(self, data: Data, params: Params) -> Data:
            sort_alphabetically = params["sort_alphabetically"]
            df = self._transform(params, pd.read_sql_query)["df"]
            if sort_alphabetically:
                df = df.sort_index(axis=1)

            return {"df": df}

        def transform_dask(self, data: Data, params: Params) -> Data:
            sort_alphabetically = params["sort_alphabetically"]
            df = self._transform(params, dd.read_sql_table)["df"]
            if sort_alphabetically:
                df = df.sort_index(axis=1)

            return {"df": df}
