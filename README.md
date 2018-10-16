# cBioF
The computational Biology Framework for initial data analysis

## Introduction
Biomedical Engineers benefit from help at the start of their data analysis. This package provides such help, by helping it with exploration, preprocessing and initial analysis. All of these processes help guiding the biomedical engineer in the direction he should go with its data analysis.

## Installation
For this package an [Anaconda](https://www.anaconda.com/download/) environment is required, using [Python](https://www.python.org/downloads/) version 3 and the packages *numpy*, *scipy*, *scikit-learn* and *pandas* that are available in Anaconda.

Two separate packages are also used, [metalearn](https://github.com/byu-dml/metalearn) and [TPOT](https://github.com/EpistasisLab/tpot). Since specific changes are made to these packages, they are already added to the framework and **do not need to be installed**. TPOT however does also use the packages *deap*, *update_checker*, *tqdm* and *stopit*. These packages have to be installed separately in Python.

To have everything correctly working, first install Anaconda. Then open the Anaconda prompt and use the following line:
```
conda install numpy scipy scikit-learn pandas matplotlib
```
Next the TPOT additional packages need to be installed with the following line in command prompt:
```
pip install deap update_checker tqdm stopit
```
Also, for windows users the package *pywin32* is needed. This is automatically installed when using Anaconda.

## Usage
Since the package itself is not possible to install yet, three different ways of using it are available:
- The first option only involves a simple GUI made for easy access of the most basic methods in the cBioF framework (discussed in the section GUI). This GUI can be started with a command prompt, by first going to the directory of the cBioF framework and then calling the file `GUI_cBioF.py` with python (using python 3 as standard version):
```
cd YOURPATH\cBiof
python GUI_cBioF.py
```
- The second way is done by using a python interpreter, by working directly in the main directory of this cBioF framework. In this main directory only the line `import cBioF` in a python file is enough to gain access to the initial cBioF methods. 
- The third option is to work in a python file in a separate directory. This directory can then be imported with the following python lines:
```\
import sys
sys.path.append(r"YOURPATH\cBioF")
import cBioF
```
How the methods should be used is explained in the following sections. Examples of the methods used are also available in the framework.

## cBioF methods
cBioF has some initial methods. These methods are based on reading in a data csv file, finding the issues in the dataset and possibly preprocessing them. After reading in the file with `read_csv_dataset`, the method `explore_dataset` forms the basics, combining both the exploration and preprocessing if also desired. For exploration the extended package *metalearn* computes metafeature values for the dataset, giving an overview over it. For preprocessing, those metafeature values are used to modify the dataset for it to become usable and improved. Dataset issues that are discussed are:

- feature types
- feature dimensionality,
- missing values 
- output imbalance
- irrelevant features
- normalisation
- multicollinearity

The newly preprocessed dataset can be exported again as a preprocessed datset with the method `export_csv_dataset`. Aside from these three methods, the method `analyse_dataset` gives a basic version of TPOT to find the best machine learning pipeline for the dataset. In this method, exploration and preprocessing is also possible to be automated, making it possible to explore, preprocess and analyse with one method. Examples are made also provided in a separate folder to show how the basic methods of the framework should be used.

### Example 1 (read_csv_dataset, explore_dataset and export_csv_dataset)

```
import cBioF

# Read the csv file. File should be a matrix with: 1st row = features, last column = output values. Input should be the path and name of a csv file
X, y, features = cBioF.read_csv_dataset('MicroOrganismsDataset.csv')

# Explore the dataset with input matrix *X*, output array *y* and feature array *features*. 
# *classification* should be True if the output values are categorical, 
# *preprocessing* should be True if also preprocessing should be done, 
# *focus* should be True if also outliers for several issues should be shown and 
# *plots* should be True if also additional plots of distributions should be shown.
X_new, y_new, f_new, exploration_results = cBioF.explore_dataset(X, y, features, output_categorical=True,
                                                                 preprocessing=True, focus=False, plots=False)
                                                                 
# Export the newly preprocessed values of the dataset
cBioF.export_csv_dataset(X_new, y_new, f_new, csv_path='HepatitisDatasetNew.csv')
```

### Example 2 (read_csv_dataset and analyse_dataset)

```
import cBioF

# Read the csv file. File should be a matrix with: 1st row = features, last column = output values. Input should be the path and name of a csv file
X, y, features = cBioF.read_csv_dataset('MicroOrganismsDataset.csv')

# Analyse the dataset with input matrix *X*, output array *y* and feature array *features*.
# *file_name* is the name of the file that will contain the best possible pipeline, as well as imports for that pipeline
# *preprocessing* should be True if preprocessing must be done beforehand
# *feature_selection* should be True if you want TPOT to focus on feature selection. This will be overridden by preprocessing
# *classification* should be True if you have classification value
cBioF.analyse_dataset(X, y, features, file_name='TPOT_pipeline_MO', preprocessing=False, feature_selection=True,
                      classification=True)
```

## GUI
The GUI makes use of the four cBioF methods made available: `read_csv_dataset`, `explore_dataset`, `analyse_dataset` and `export_csv_dataset`. For all of these important input variables can be changed by the user. Progress can be tracked by reading the terminal provided in the window. Two screenshots:

- The starting screen of the GUI
![alt text](https://github.com/TimBeishuizen/cBioF/blob/master/GUI_pictures/example_GUI.PNG "Starting screen cBioF GUI")

- The GUI doing exploration and preprocessing of a dataset
![alt text](https://github.com/TimBeishuizen/cBioF/blob/master/GUI_pictures/example_GUI_2.PNG "Usage GUI")

## Builtin functions
Aside from the exploration and preprocessing, several methods are also given separately. These methods are made when the user has a better idea how to approach the dataset and preprocess it. These methods are mainly created due to limitations in the *scikit-learn* package and alternatives usually can be found in there. The builtin functions are based on feature selection, missing value handling and basic preprocessing. All of them need numpy matrices or arrays as input, either from the data matrix, the output array or the feature array. Explaining every single method and parameter is done in the code. An explanation of a method `TARGET_METHOD` can be retrieved by putting the line `help(TARGET_METHOD)` in Python after importing it.

### Feature selection
Feature selection methods can be used when using the python line `import cBioF.feature_selection`. Feature selection algorithms can be divided into three categories: filter, wrapper and embedded methods. Of these three, filter methods and embedded methods are mostly provided by the package *scikit-learn*. Wrapper methods, however, are not and therefore are provided in this framework in the same stile as the other methods in *scikit-learn*.
- Wrapper methods: Forward selection (`ForwardSelector`), backward selection (`BackwardSelector`), PTA (`PTA`) and floating selection (`FloatingSelector`)
- Embedded methods: K-from-model selection (select K best instead of choosing a model specific threshold, `SelectKFromModel`)

### Missing value handling
Missing value handling methods can be used when using the python line `import cBioF.missing_value_handling`.Some missing value handling algorithms can be found in *scikit-learn*. These however, do have the condition of all features being numeric and are only basic imputation methods. Misisng value handling is the first step done for a dataset, as other methods usually require a complete dataset. Therefore missing value handling methods are given that also accept categorical data.
- List Deletion methods: Complete case analysis (`cca`), available case analysis (`aca`)
- Imputation methods: Mean imputation (`mean_imputation`), hot deck imputation (`hot_deck_imputation`), missing indicator imputation (`missing_indicator_imputation`), regression imputation (`regression_imputation`), k-nearest neighbour imputation (`kNN_imputation`)

### Preprocessing
Preprocessing methods can be used when using the python line `import cBioF.preprocessing`.Two important approaches for categorical data and numeric data are hot encoding and normalisation respectively. Hot encoding must be done when categorical data is analysed using numeric data analyses. Normalisation must be done to remove possible bias created by the data distributions. For both methods in *scikit-learn* are available. These methods however are either not able to hot encode textual categorical data or not guiding the user in to using the proper normalisation techniques. Therefore for both a new method was created:
- Hot encoding (`hot_encoding`)
- Normalisation (`normalisation`)

## Automated Machine learning
 One way of analysing the data is using machine learning. Machine learning predicts future behaviour of a phenomenon by using known data. There are many different machine learning techniques however and choosing the best one might be difficult. Therefore added to the framework is the package *TPOT* that implements automated machine learning. Automated machine learning automatically select the best possible techniques and also includes several preprocessing techniques. An extension was made for *TPOT* for datasets with many features for improved feature selection search during the automated machine learning.

### TPOT
TPOT can be used when using the python line `import cBioF.TPOT_extension`. TPOT is an implementation of automated machine learning. It uses an evolution algorithm to find the best combination of machine learning and preprocessing methods. Several parameters can be changed for making the search as extensive as desired, such as population size, generation size and search time.

#### Feature selection TPOT
When the dataset has a high number of features, a special type of TPOT can be chosen. Simply the parameter *feature_selection* should be set on *True*. With that a different type of accuracy is used that adds a penalty on the number of features to the outcome. On top of that two changes can be chosen:

- Focus on feature selection: Have the first generation all start with a feature selection algorithm to quicken the process
- Change of feature selection algorithms: Change the feature selection algorithm to a set that focuses on keeping at most 200 features.

With these two additions a more effective search can be done to a good data analysis pipeline.

