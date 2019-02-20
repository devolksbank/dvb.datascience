from typing import List

from ..data import MetaData
from ..sub_pipe_base import SubPipelineBase
from ..transform import (
    CategoricalImpute,
    FilterFeatures,
    ImputeWithDummy,
    LabelBinarizerPipe,
    LabelEncoderPipe,
    Union,
)


class MetadataPipeline(SubPipelineBase):
    """
    Read metadata and make some pipes for processing the data
    """

    input_keys = ('df',)
    output_keys = ("df",)

    def __init__(self, metadata: MetaData, remove_features: List = None, keep_features: List = None) -> None:
        super().__init__('unknown')

        self.metadata = metadata
        self.remove_features = remove_features
        self.keep_features = keep_features

        if self.keep_features is not None:
            rows = [
                row
                for row in self.metadata.vars.values()
                if row['varName'] in self.keep_features
            ]
        elif self.remove_features is not None:
            rows = [
                row
                for row in self.metadata.vars.values()
                if row['varName'] not in self.remove_features
            ]
        else:
            # all features as specified in metadata are used
            rows = self.metadata.vars.values()

        self._last_pipe_name = "pass_data"

        def concat_pipe(name, pipe):
            self.sub_pipeline.addPipe(
                name, pipe, [(self._last_pipe_name, "df", "df")]
            )
            self._last_pipe_name = name

        for idx, row in enumerate(rows):
            if row['impMethod'] == "none" and row['impValue'] != float("nan"):
                concat_pipe(
                    "impute_" + row['varName'],
                    ImputeWithDummy(strategy="value", value=row['impValue'], features=[row['varName']]),
                )

            elif row['varType'] in ("numi", "numc"):
                concat_pipe(
                    "impute_" + row['varName'],
                    ImputeWithDummy(strategy=row['impMethod'], features=[row['varName']]),
                )
            elif row['varType'] == "ind":
                # all boolean field already have -1 for missing values by read_csv
                pass
            if row['varType'] in ("cat" , "cd", "oms") and row['impMethod'] == "mode":
                concat_pipe(
                    "impute_" + row['varName'],
                    CategoricalImpute(features=[row['varName']]),
                )
                concat_pipe(
                    "labelencoder_" + row['varName'],
                    LabelEncoderPipe(features=[row['varName']]),
                )

            elif row['varType'] in ("cat" , "cd", "oms") :
                concat_pipe(
                    "labelencoder_" + row['varName'],
                    LabelEncoderPipe(features=[row['varName']]),
                )

        self.setOutputPipeName(self._last_pipe_name)
