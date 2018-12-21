import codecs
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with codecs.open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dvb.datascience",  # Required
    version="0.13.dev0",
    description="Some helpers for our data scientist",  # Required
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/devolksbank/dvb.datascience",  # Optional
    author="Technology Center, de Volksbank (NL)",  # Optional
    author_email="tc@devolksbank.nl",  # Optional
    license="MIT License",
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=True,
    keywords="datascience sklearn pipeline pandas eda train",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
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
    ],  # Optional
    extras_require={  # Optional
        "dev": ["zest.releaser[recommended]"],
        "test": ["coverage", "pytest", "pytest-cov", "pexpect"],
        "release": ["zest.releaser"],
        "teradata": ["teradata"],
        "docs": ["sphinx", "m2r", "nbsphinx", "jupyter_client", "nbconvert==5.3.1"],
    },
    package_data={},  # Optional
    data_files=[],  # Optional
    entry_points={},  # Optional
    project_urls={},  # Optional
)
