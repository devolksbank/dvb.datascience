import pytest

import dvb.datascience as ds

@pytest.mark.usefixtures("dataframe_engine")
class TestInitMethods:
    @pytest.mark.skip()
    def test_run_module(self):
        ds.run_module("score_test_script").run()
