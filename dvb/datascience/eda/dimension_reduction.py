import logging

import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn import manifold
from sklearn.decomposition import PCA

from ..pipe_base import Data, Params
from .base import AnalyticsBase

logger = logging.getLogger(__name__)


class DimensionReductionPlots(AnalyticsBase):
    """
    Plot dimension reduction graphs of the data.

    Args:
        data: Dataframe to be used.

    Returns:
        Dimension reduction (PCA, ISOMAP, MDS, Spectral Embedding, tSNE) plots of the data
    """

    input_keys = ("df",)
    output_keys = ("figs",)

    def __init__(self, y_label):
        super().__init__()

        self.y_label = y_label
        self.n_neighbors = 10
        self.n_components = 2
        self.pca = None

    def transform_pandas(self, data: Data, params: Params) -> Data:
        df = data["df"]
        if self.y_label not in df.columns:
            display(
                "y_label from init [%s] not present in dataframe columns [%s]"
                % (self.y_label, df.columns)
            )
            return {}

        y = df[self.y_label]
        X = df.drop(self.y_label, axis=1)

        # PCA
        self.pca = PCA(n_components=self.n_components)
        Xt = self.pca.fit_transform(X, y)
        self.scatterPlot(Xt, y, "PCA")

        for method in ["standard", "modified"]:  # ltsa, hessian
            Xt = manifold.LocallyLinearEmbedding(
                self.n_neighbors, self.n_components, eigen_solver="auto", method=method
            ).fit_transform(X)
            self.scatterPlot(Xt, y, method)

        # ISOMAP
        Xt = manifold.Isomap(self.n_neighbors, self.n_components).fit_transform(X)
        self.scatterPlot(Xt, y, "isomap")

        # MDS
        mds = manifold.MDS(self.n_components, max_iter=100, n_init=1)
        Xt = mds.fit_transform(X)
        self.scatterPlot(Xt, y, "MDS")

        # SE
        Xt = manifold.SpectralEmbedding(
            n_components=self.n_components, n_neighbors=self.n_neighbors
        ).fit_transform(X)
        self.scatterPlot(Xt, y, "Spectral Embedding")

        # TSNE
        Xt = manifold.TSNE(
            n_components=self.n_components, init="pca", random_state=0
        ).fit_transform(X)
        self.scatterPlot(Xt, y, "t-SNE")

        return {"figs": self.figs}

    def transform_dask(self, data: Data, params: Params) -> Data:
        raise NotImplementedError

    def scatterPlot(self, X, y, title):
        logger.info("Creating scatterplot: %s", title)
        figure = plt.figure(figsize=(15, 8))
        self.figs[title] = figure
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title(title)
        plt.axis("tight")
        display(self.figs[title])
