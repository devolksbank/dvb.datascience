# dvb.datascience

A python [data science](https://en.wikipedia.org/wiki/Data_science) pipeline package. Full documentation can be found on [ReadTheDocs](https://dvbdatascience.readthedocs.io/en/latest/)

![Travis](https://travis-ci.org/devolksbank/dvb.datascience.svg?branch=master)
[![codecov](https://codecov.io/gh/devolksbank/dvb.datascience/branch/master/graph/badge.svg)](https://codecov.io/gh/devolksbank/dvb.datascience)

At [de Volksbank](https://www.devolksbank.nl/), our data scientists used to write a lot of overhead code for every experiment from scratch. To help them focus on the more exciting and value added parts of their jobs, we created this package.
Using this package you can easily create and reuse your pipeline code (consisting of often used data transformations and modeling steps) in experiments.

![Sample Project Gif](https://dvbdatascience.readthedocs.io/en/latest/_images/GIF_Sample_Project.gif)

This package has (among others) the following features:

- Make easy-to-follow model pipelines of fits and transforms ([what exactly is a pipeline?](https://stackoverflow.com/questions/33091376/python-what-is-exactly-sklearn-pipeline-pipeline))
- Make a graph of the pipeline
- Output graphics, data, metadata, etc from the pipeline steps
- Data preprocessing such as filtering feature and observation outliers
- Adding and merging intermediate dataframes
- Every pipe stores all intermediate output, so the output can be inspected later on
- Transforms can store the outputs of previous runs, so the data from different transforms can be compared into one graph
- Data is in [Pandas](https://pandas.pydata.org/) DataFrame format
- Parameters for every pipe can be given with the pipeline fit_transform() and transform() methods

![logo](https://www.devolksbank.nl/upload/d201c68e-5401-4722-be68-6b201dbe8082_de_volksbank.png "De Volksbank - The Netherlands")


## Scope

This package was developed specifically for fast prototyping with relatively small datasets on a single machine. By allowing the intermediate output of each pipeline step to be stored, this package might underperform for bigger datasets (100,000 rows or more).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
For a more extensive overview of all the features, see the docs directory.

### Prerequisites

This package requires [Python3](https://www.python.org/) and has been tested/developed using python 3.6 and 3.7.

### Installing

The easiest way to install the library (for using it), is using:

```bash
pip install dvb.datascience
```

#### Development

(in the checkout directory): For installing the checkouts repo for developing of dvb.datascience:

```bash
pip install -e .
```

For using dvb.datascience in your project:

```bash
pip install dvb.datascience
```

#### Development - Anaconda

(in the checkout directory): Create and activate an environment + install the package:

```bash
conda create --name dvb.datascience
conda activate dvb.datascience
pip install -e .
```

or use it via:

```bash
pip install dvb.datascience
```

#### Jupyter table-of-contents

When working with longer pipelines, the output when using a jupyter notebook can become quite long. It is advisable to install the
[nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) for the [toc2](https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tree/master/src/jupyter_contrib_nbextensions/nbextensions/toc2) extension:

```bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install
```

Next, start a jupyter notebook and navigate to [edit > nbextensions config](http://localhost:8888/nbextensions/) and enable the toc2 extension. And optionally set other properties.
After that, navigate back to your notebook (refresh) and click the icon in the menu for loading the toc in the side panel.

## Examples

This example loads the data and makes some plots of the Iris dataset

```python
import dvb.datascience as ds


p = ds.Pipeline()
p.addPipe('read', ds.data.SampleData('iris'))
p.addPipe('split', ds.transform.RandomTrainTestSplit(test_size=0.3), [("read", "df", "df")])
p.addPipe('boxplot', ds.eda.BoxPlot(), [("split", "df", "df")])
p.fit_transform(transform_params={'split': {'train': True}})
```

This example shows a number of features of the package and its usage:

- Adding 3 steps to the pipeline using `addPipe()`.
- Linking the 3 steps using `[("read", "df", "df")]`: the `'df'` output (2nd parameter) of the `"read"` method (1st method) to the `"df"` input (3rd parameter) of the split method.
- The usage of 3 subpackages: `ds.data`, `ds.transform` and `ds.eda`. The other 2 packages are: `ds.predictor` and `ds.score`.
- The last method `p.fit_transform()` has as a parameter additional input for running the defined pipeline, which can be different for each call to the `p.fit_transform()` or `p.transform()` method.

This example applies the KNeighborsClassifier from sklearn to the Iris dataset

```python
import dvb.datascience as ds

from sklearn.neighbors import KNeighborsClassifier
p = ds.Pipeline()
p.addPipe('read', ds.data.SampleData('iris'))
p.addPipe('clf', ds.predictor.SklearnClassifier(KNeighborsClassifier, n_neighbors=3), [("read", "df", "df"), ("read", "df_metadata", "df_metadata")])
p.addPipe('score', ds.score.ClassificationScore(), [("clf", "predict", "predict"), ("clf", "predict_metadata", "predict_metadata")])
p.fit_transform()
```

This example shows:

- The use of the `KNeighborsClassifier` from `sklearn`
- The usage of coupling of multiple parameters as input: `[("read", "df", "df"), ("read", "df_metadata", "df_metadata")]`

For a more extensive overview of all the features, see the docs directory.

## Backends

Both Pandas and Dask can be used for the actual processing. At default Pandas will be used. When needed, Dask can be
used to handle datasets which do not fit in memory or when distributed computing is needed. Dask does is not a drop-in
replacement of Pandas; some methods are not provided by Dask. Like computing the median can not be distributed efficiently
and therefor is [not implemented in Dask](http://docs.dask.org/en/latest/dataframe.html).

## Unittesting

The unittests for the project can be run using [pytest](https://pytest.org/):

```bash
pytest
```

All tests are executed with both the Pandas and Dask engine.

### Code coverage

Pytest will also output the coverage tot the console.

To generate an html report, you can use:

```bash
py.test --cov-report html
```

## Code styling

Code styling is done using [Black](https://pypi.org/project/black/)

## Built With

For an extensive list, see [setup.py](setup.py)

- [scipy / numpy / pandas / matplotlib](https://www.scipy.org/) - For calculations and visualizations
- [sklearn](http://scikit-learn.org/stable/) - Machine learning algorithms
- [statsmodels](https://www.statsmodels.org/stable/index.html) - Statistics
- [mlxtend](https://rasbt.github.io/mlxtend/) - Feature selection
- [tabulate](https://pypi.org/project/tabulate/) - Printing tabular data
- [imblearn](https://pypi.org/project/imblearn/) - SMOTE

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/devolksbank/dvb.datascience/tags).

## Authors

- **Marc Rijken** - _Initial work_ - [mrijken](https://github.com/mrijken)
- **Wouter Poncin** - _Maintenance_ - [wpbs](https://github.com/wpbs)
- **Daan Knoope** - _Contributor_ - [daanknoope](https://github.com/daanknoope)
- **Christopher Huijting** - _Contributor_ - [chuijting](https://github.com/chuijting)

See also the list of [contributors](https://github.com/devolksbank/dvb.datascience/blob/master/CONTRIBUTORS) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Contact

For any questions please don't hesitate to contact us at [tc@devolksbank.nl](mailto:tc@devolksbank.nl)

## Work in progress

- Adding support for multiclass classification problems
- Adding support for regression problems
- Adding support for Apache Spark ML
