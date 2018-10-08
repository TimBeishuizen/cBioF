# cBioF
The computational Biology Framework for initial data analysis

## Introduction
Biomedical Engineers benefit from help at the start of their data analysis. This package provides such help, by helping it with exploration, preprocessing and initial analysis. All of these processes help guiding the biomedical engineer in the direction he should go with its data analysis.

## Installation
For this package an [Anaconda](https://www.anaconda.com/download/) environment is advised, using [Python](https://www.python.org/downloads/) version 3 and the available packages *numpy*, *scipy*, *scikit-learn* and *pandas* that are available in Anaconda.

Two separate packages are also used, [metalearn](https://github.com/byu-dml/metalearn) and [TPOT](https://github.com/EpistasisLab/tpot). Since specific changes are made to these packages, they are separately added to the framework. TPOT however does also use the packages *deap*, *update_checker*, *tqdm* and *stopit*. These packages have to be installed separately in Python.

To have everything correctly working, first install Anaconda. Then open Anaconda prompt and say the following:
- conda install numpy scipy scikit-learn pandas<br /> <br />
Next the TPOT additional packages need to be installed:
- pip install deap update_checker tqdm stopit<br /> <br />
After this, the framework should be ready for usage.

## cBioF methods
cBioF has some initial methods. These methods are based on reading in a data csv file, finding the issues in the dataset and possibly preprocessing them. After reading in the file with *read_csv_dataset*, the method *find_dataset_issues* forms the basics, combining both the exploration and preprocessing. For exploration the extended package *metalearn* computes metafeature values for the dataset, giving an overview over it. For preprocessing, those metafeature values are used to modify the dataset for it to become usable and improved. Dataset issues that are discussed are:

- feature types
- feature dimensionality,
- missing values 
- output imbalance
- irrelevant features
- normalisation
- multicollinearity

## Builtin functions
Aside from the exploration and preprocessing, several methods are also given separately. These methods are made when the user has a better idea how to approach the dataset and preprocess it. These methods are mainly created due to limitations in the *scikit-learn* package and alternatives usually can be found in there. The builtin functions are based on feature selection, missing value handling and basic preprocessing. All of them need numpy matrices or arrays as input, either from the data matrix, the output array or the feature array.

### Feature selection
Feature selection algorithms can be divided into three categories: filter, wrapper and embedded methods. Of these three, filter methods and embedded methods are mostly provided by the package *scikit-learn*. Wrapper methods, however, are not and therefore are provided in this framework in the same stile as the other methods in *scikit-learn*.
- Wrapper methods: Forward selection, backward selection, PTA and floating selection
- Embedded methods: Select K from model (select K best instead of choosing a percentage)

### Missing value handling
Some missing value handling algorithms can be found in *scikit-learn*. These however, do have the condition of all features being numeric and are only basic imputation methods. Misisng value handling is the first step done for a dataset, as other methods usually require a complete dataset. Therefore missing value handling methods are given that also accept categorical data.
- List Deletion methods: cca, aca
- Imputation methods: Mean imputation, hot deck imputation, missing indicator imputation, regression imputation, kNN imputation)

### Preprocessing
Two important approaches for categorical data and numeric data are hot encoding and normalisation respectively. Hot encoding must be done when categorical data is analysed using numeric data analyses. Normalisation must be done to remove possible bias created by the data distributions. For both methods in *scikit-learn* are available. These methods however are either not able to hot encode textual categorical data or not guiding the user in to using the proper normalisation techniques. Therefore for both a new method was created:
- Hot encoding
- Normalisation

## Machine learning
One way of analysing the data is using machine learning. Machine learning predicts future behaviour of a phenomenon by using known data. There are many different machine learning techniques however and choosing the best one might be difficult. Therefore added to the framework is the package *TPOT* that implements automated machine learning. Automated machine learning automatically select the best possible techniques and also includes several preprocessing techniques. An extension was made for *TPOT* for datasets with many features for improved feature selection search during the automated machine learning.

### TPOT
TPOT is an implementation of automated machine learning. It uses an evolution algorithm to find the best combination of machine learning and preprocessing methods. Several parameters can be changed for making the search as extensive as desired, such as population size, generation size and search time.

### Feature selection TPOT
When the dataset has a high number of features, a special type of TPOT can be chosen. Simply the parameter *feature_selection* should be set on *True*. With that a different type of accuracy is used that adds a penalty on the number of features to the outcome. On top of that two changes can be chosen:

- Focus on feature selection: Have the first generation all start with a feature selection algorithm to quicken the process
- Change of feature selection algorithms: Change the feature selection algorithm to a set that focuses on keeping at most 200 features.

With these two additions a more effective search can be done to a good data analysis pipeline.

