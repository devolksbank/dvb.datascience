import pytest

from dvb.datascience import pipeline

@pytest.fixture(params=['pandas', 'dask'], )
def dataframe_engine(request):
    if request.node.get_closest_marker('skip_dataframe_engine'):
        if request.node.get_closest_marker('skip_dataframe_engine').args[0] == request.param:
            pytest.skip('skipped on this dataframe: {}'.format(request.param))
            return
    pipeline.default_dataframe_engine = request.param