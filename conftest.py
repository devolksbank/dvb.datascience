import pytest

from dvb.datascience import pipeline

def pytest_addoption(parser):
    parser.addoption(
        "--dataframe-engine", action="store", default="pandas", help="Specify the dataframe engine"
    )

def pytest_configure(config):
    print(config.getoption("--dataframe-engine"))

    pipeline.default_dataframe_engine = config.getoption("--dataframe-engine")