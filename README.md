# cBioF
The computational Biology Framework for initial data analysis

## Introduction
Biomedical Engineers benefit from help at the start of their data analysis. This package provides such help, by helping it with exploration, preprocessing and initial analysis. All of these processes help guiding the biomedical engineer in the direction he should go with its data analysis.

## Installation
For this package an [Anaconda 3](https://www.anaconda.com/download/) environment is advised, using [Python 3](https://www.python.org/downloads/) and the available packages numpy, scipy, scikit-learn and pandas that are available in Anaconda.

Two separate packages are also used, [metalearn](https://github.com/byu-dml/metalearn) and [TPOT](https://github.com/EpistasisLab/tpot). Since specific changes are made to these packages, they are separately added to the framework. TPOT however does also use the packages deap, update_checker, tqdm and stopit. These packages have to be installed separately in Python.

To have everything correctly working, first install anaconda 3. Then open anaconda prompt and say the following:
- conda install numpy scipy scikit-learn pandas
Next the TPOT additional packages need to be installed:
- pip install deap update_checker tqdm stopit
After this, the framework should be ready for usage.

## cBioF methods

- Create first impressions
- using metalearn methods (link to metalearn)

### Exploration
- Exploration on these topics...

### Preprocessing
- Preprocessing on explored topics...

## Builtin functions

### Feature selection
- Based on aggressive feature selection
- Wrapper methods
- SelectKfromModel

### Missing value handling
- Based on first step
- List Deletion methods (CCA, ACA)
- Imputation methods (Mean, hot deck, regression, missing indicator, kNN, regression)

### Preprocessing
- hot encoding
- normalisation

## Machine learning

- automated machine learning
- using TPOT (link to TPOT)

### Regular TPOT
- algorithms
- TPOT possibilities

### Feature selection TPOT
- Algorithms
- TPOT possibilities
