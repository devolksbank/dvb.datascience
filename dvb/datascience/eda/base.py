import matplotlib.pyplot as plt
from typing import Any, Dict

from ..pipe_base import Data, Params, PipeBase


class AnalyticsBase(PipeBase):
    def __init__(self):
        super().__init__()

        self.number_of_dfs = 0
        self._reset_figs()

    def fit(self, data: Data, params: Params):
        self._reset_figs()

    def _reset_figs(self):
        self.figs: Dict[Any, plt.Figure] = {}
        self.number_of_dfs = 0

    def set_fig(self, idx: Any):
        """
        Set in plt the figure to one to be used.

        If 'idx' has already been used, the data will be added to the plot used with this idx.
        If not, a new figure will be created.
        """

        if idx not in self.figs:
            self.figs[idx] = plt.figure()

        fig = self.figs[idx]
        plt.figure(fig.number)

    def get_number_of_dfs(self):
        self.number_of_dfs += 1
        return self.number_of_dfs
