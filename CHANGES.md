# CHANGES

## 0.13 (unreleased)

- Refactor storage of pipe outputs with optional saving them to file
- Add Dask as dataframe engine, so tasks can be distributes among multiple cores and/or machines


## 0.12 (2018-09-19)

- Public Release

## 0.11 (2018-09-10)

- Improvements made on skipping pipes which are not connected
- Minor fixes
- Improve Union
- Documentation improvements

## 0.10 (2018-07-30)

- Replaced mermaid-cli with blockdiag
- Restructured unittests and prepared code coverage reporting
- Added unittest cases
- Added notes on pipe's for displaying additional information in blockdiag


## 0.9 (2018-07-25)

- Fixed eda.Describe()
- Added eda.Andrews
- Added eda.Dump()
- Added example notebooks in docs
- Several code quality improvements


## 0.8 (2018-07-24)


- Add hyperparameters for logit summary
- Add option to skip a pipeline step in transform step
- Add GridSearch
- Improve histogram
- Improve AUC plot
- Fixed unittests


## 0.7 (2018-06-26)


- Disable the part of model performance which is very slow
- Add some convenience code for running experiments from notebooks
- Add option to disable the splittig based on transform parameters
- Update logging


## 0.6.1 (2018-06-22)

- Add timing

## 0.6 (2018-06-21)

- Add feature for importing Excel file
- Add feature to RemoveOutliers to remove only observation with a given nr of outliers
- Add SMOTE

## 0.5.5 (2018-05-29)

- Make generating drawings of the pipeline optional

## 0.5.4 (2018-05-24)

- Add impute by mode and value

## 0.5.3 (2018-05-24)

- Add orginal column name to binarized column


## 0.5.2 (2018-05-24)

- Fix release error

## 0.5.1 (2018-05-24)

- Add skip_columns to RemoveOutliers

## 0.5 (2018-05-24)

- Add removal of observations based on outliers

## 0.4.1 (2018-05-23)

- bugfix in logit summary (when runned without label)
- bugfix in conf.matrix (do not restrict the number of features)

## 0.4 (2018-05-14)


- Keep all outputs of all transforms in the pipeline
- Add correlatio matrix
- Add Drop Correlated Features
- Add Logistic Summary
- Add Swarm Plots
- Make a graph with Mermaid at every fit
- Several bug fixes

## 0.3.1 (2018-05-09)

- Some smalle fixes

## 0.3 (2018-05-08)

- Improve scoring methods
- Some bugfixes
- Add option to connect to other pipes from addPipe

## 0.2 (2018-05-03)

- Add metadata parser
- Add subpipeline
- Some bug fixes

## 0.1 (2018-05-01)

- First release