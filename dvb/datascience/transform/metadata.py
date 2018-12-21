from typing import List

from ..data import MetaData
from ..sub_pipe_base import SubPipelineBase
from ..transform import (
    CategoricalImpute,
    FilterFeatures,
    ImputeWithDummy,
    LabelBinarizerPipe,
    Union,
)


class MetadataPipeline(SubPipelineBase):
    """
    Read metadata and make some pipes for processing the data
    """

    input_keys = ('df',)
    output_keys = ("df",)

    def __init__(self, metadata: MetaData, remove_vars: List = None) -> None:
        super().__init__("union")

        self.remove_vars = remove_vars or []
        self.metadata = metadata

        rows = [
            row
            for row in self.metadata.vars.values()
            if row['nameVar'] not in self.remove_vars
        ]

        union_input = []

        for idx, row in enumerate(rows):
            if idx == 10:  # TODO: Remove
                break

            self.sub_pipeline.addPipe(
                "filter_" + row['nameVar'], FilterFeatures([row['nameVar']]),
                [("pass_data", "df", "df")]
            )

            if row['varType'] == "numi" and row['impMethod'] in ["mean", "median"]:
                self.sub_pipeline.addPipe(
                    "impute_" + row['nameVar'], ImputeWithDummy(strategy=row['impMethod']),
                    [("filter_" + row['nameVar'], "df", "df")]
                )
                union_input.append(("impute_" + row['nameVar'], idx))

            elif row['varType'] == "cat" and row['impMethod'] == "mode":
                self.sub_pipeline.addPipe(
                    "impute_" + row['nameVar'],
                    CategoricalImpute(),
                    [("filter_" + row['nameVar'], "df", "df")]
                )
                self.sub_pipeline.addPipe(
                    "labelbinarizer_" + row['nameVar'],
                    LabelBinarizerPipe(),
                    [("impute_" + row['nameVar'], "df", "df")]
                )
                union_input.append(("labelbinarizer_" + row['nameVar'], idx))

            elif row['varType'] == "cat":
                self.sub_pipeline.addPipe(
                    "labelbinarizer_" + row['nameVar'],
                    LabelBinarizerPipe(),
                    [("filter_" + row['nameVar'], "df", "df")]
                )
                union_input.append(("labelbinarizer_" + row['nameVar'], idx))
            else:
                union_input.append(("filter_" + row['nameVar'], idx))

        self.sub_pipeline.addPipe(
            "union",
            Union(len(rows)),
            [(i[0], "df", "df%s" % i[1]) for i in union_input]
        )

