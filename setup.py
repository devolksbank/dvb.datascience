import codecs
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dvb.datascience",
    version="0.13.dev0",
    description="Some helpers for our data scientist",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devolksbank/dvb.datascience",
    author="Technology Center, de Volksbank (NL)",
    author_email="tc@devolksbank.nl",
    license="MIT License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=True,
    keywords="datascience sklearn pipeline pandas eda train",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "statsmodels",
        "mlxtend",
        "tabulate",
        "imblearn",
        "seaborn",
        "patsy",
        "blockdiag",
        "pyensae",
        "TPOT",
        "xlrd",
        "dask[array, bag, dataframe, distributed, delayed]",
        "dask_ml",
    ],
    extras_require={
        "dev": ["zest.releaser[recommended]"],
        "test": ["coverage", "pytest>=4.0", "pytest-cov", "pexpect"],
        "release": ["zest.releaser"],
        "teradata": ["teradata"],
        "docs": ["sphinx", "m2r", "nbsphinx", "jupyter_client", "nbconvert==5.3.1"],
    },
    package_data={},
    data_files=[],
    entry_points={},
    project_urls={},
)
