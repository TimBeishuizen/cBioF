import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cBioF import preprocess_dataset as RDIM
from cBioF.metalearn_extension import Metafeatures
from cBioF.robustness_methods import robustness_methods


def explore_dataset(X, y, features, missing_values='Unknown', preprocessing=False, classification=True,
                        focus=False, plots=False):
    """ First prepare the data for data exploration. Then find issues in the dataset. These issues are based on:
     - feature types, 
     - feature dimensionality, 
     - missing values, 
     - output imbalance,
     - irrelevant features,
     - normalisation,
     - multicollinearity
     
     The exploration can include a more thorough focus on outliers. Also several plots can be shown to give more insight
     in the dataset.
     If desired the output can also be preprocessed and returned immediately, with and without feature selection.

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param y: A numpy array of the output. The length of the array should correspond to the size of the first 
    axis of X
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the 
    second axis of X
    :param missing_values: The way missing values are added to the data. Can be a list if missing values are portrayed 
    in multiple ways. Default: 'Unknown' (several well-known ways of portraying missing values are checked)
    :param preprocessing: Whether preprocessing should be done after exploration
    :param classification: If the output is categorical. Default: True
    :param focus: Whether focus is desired on outliers to get a better insight in the data. Default: False
    :param plots: Whether plots should be shown to get a better insight in the data. Default: False
    :return: if: preprocessing = True: preprocessed X, y and features
    """

    # Test the input to be according to the standards
    robustness_methods.check_input_arrays(X, y, features)

    dfX, dfy = _create_pandas_dataframe_dataset(X, y, features, classification=classification)

    dfX, dfy, exploration_results = pandas_explore_dataset(dfX, dfy, missing_values=missing_values,
                                                               preprocessing=preprocessing, focus=focus,
                                                               plots=plots)
    X_new = dfX.values
    y_new = dfy.values
    f_new = list(dfX)

    return X_new, y_new, f_new, exploration_results


def pandas_explore_dataset(dfX, dfy, missing_values='Unknown', preprocessing=False, focus=False, plots=False):
    """ First prepare the data for data exploration. Then find issues in the dataset. These issues are based on:
     - feature types, 
     - feature dimensionality, 
     - missing values, 
     - output imbalance,
     - irrelevant features,
     - normalisation,
     - multicollinearity
     
     The exploration can include a more thorough focus on outliers. Also several plots can be shown to give more insight
     in the dataset.
     If desired the output can also be preprocessed and returned immediately, with and without feature selection.

    :param dfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :param dfy: A pandas series of the output. The length of the array should correspond to the size of the first 
    axis of X
    :param missing_values: The way missing values are added to the data. Can be a list if missing values are portrayed 
    in multiple ways. Default: 'Unknown' (several well-known ways of portraying missing values are checked)
    :param preprocessing: Whether preprocessing should be done after exploration
    :param focus: Whether focus is desired on outliers to get a better insight in the data. Default: False
    :param plots: Whether plots should be shown to get a better insight in the data. Default: False
    :return: if: preprocessing = True: preprocessed dfX and dfy
    """

    # Test the input to be according to the standards
    robustness_methods.check_pandas_input(dfX, dfy)

    dfX = _prepare_dataset(dfX, missing_values=missing_values)

    exploration_results = _pandas_metalearn_explore_dataset(dfX, dfy, plots=plots, focus=focus)

    # If preprocessing is needed
    if preprocessing:
        print("\n \033[94m Preprocessing \033[m")
        dfX_new, dfy_new = RDIM.pandas_preprocess_dataset(dfX, dfy, exploration_results, fs_example=True)

        return dfX_new, dfy_new, exploration_results


def _pandas_metalearn_explore_dataset(dfX, dfy, focus=False, plots=False):
    """ TFind issues in the dataset. These issues are based on:
     - feature types, 
     - feature dimensionality, 
     - missing values, 
     - output imbalance,
     - irrelevant features,
     - normalisation,
     - multicollinearity
     
     The exploration can include a more thorough focus on outliers. Also several plots can be shown to give more insight
     in the dataset.

    :param dfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :param dfy: A pandas series of the output. The length of the array should correspond to the size of the first 
    axis of X
    :param focus: Whether focus is desired on outliers to get a better insight in the data. Default: False
    :param plots: Whether plots should be shown to get a better insight in the data. Default: False
    :return: if: preprocessing = True: preprocessed dfX and dfy
    """

    # Test the input to be according to the standards
    robustness_methods.check_pandas_input(dfX, dfy)

    mf = Metafeatures()
    metafeatures = mf.compute(dfX, dfy)


    exploration_results = {}
    exploration_results['cat'] = False
    exploration_results['num'] = False

    # Feature types
    if metafeatures['NumberOfCategoricalFeatures']['value'] > 0:
        print("\n \033[94m Feature Types\033[0m")
        exploration_results['cat'] = True
        print("%i categorical features are present, hot encoding is recommended for machine learning analysis"
              % metafeatures['NumberOfCategoricalFeatures']['value'])
        if metafeatures['NumberOfNumericFeatures']['value'] > 0:
            exploration_results['num'] = True
            print("Also %i numerical features are present. Either hot encoding or binning must occur for "
                  "further analysis" % metafeatures['NumberOfNumericFeatures']['value'])
    elif metafeatures['NumberOfNumericFeatures']['value'] > 0:
        exploration_results['num'] = True



    # Feature dimensionality
    exploration_results['fs'] = False

    if metafeatures['Dimensionality']['value'] > 1:
        print("\n \033[94m Feature Dimensionality\033[m")
        exploration_results['fs'] = True
        print("Dimensionality is higher than 1: %.2f, feature selection is recommended"
              % metafeatures['Dimensionality']['value'])

    min_mi = min(metafeatures['MinCategoricalMutualInformation']['value'],
                 metafeatures['MinNumericMutualInformation']['value'])

    if min_mi < 0.01:
        exploration_results['fs'] = True
        print("The mutual information of at least one feature is lower than 0.05: %.2f. "
              "Feature selection is recommended" % min_mi)

    exploration_results['mv'] = False
    exploration_results['cca'] = True
    exploration_results['aca'] = True

    if focus and exploration_results['cat']:
        feat = metafeatures['CategoricalMutualInformationOutlierFeatures']['value']
        out = [round(elem, 2) for elem in metafeatures['CategoricalMutualInformationOutliers']['value']]
        print("The categorical features %s have the lowest mutual information respectively %s" %
              (feat[0:3], out[0:3]))
        print("The categorical features %s have the highest mutual information respectively %s" %
              (feat[3:6], out[3:6]))
        feat = metafeatures['NumericMutualInformationOutlierFeatures']['value']
        out = [round(elem, 2) for elem in metafeatures['NumericMutualInformationOutliers']['value']]
        print("The numeric features %s have the lowest mutual information respectively %s" %
              (feat[0:3], out[0:3]))
        print("The numeric features %s have the highest mutual information respectively %s" %
              (feat[3:6], out[3:6]))

    # Missing values
    if metafeatures['NumberOfMissingValues']['value'] > 0:
        print("\n \033[94m Missing Values \033[0m")
        exploration_results['mv'] = True
        print("%i missing values are present, missing value handling should occur"
              % metafeatures['NumberOfMissingValues']['value'])
        if metafeatures['RatioOfInstancesWithMissingValues']['value'] > 0.10:
            print("The ratio of samples with missing values is not small %.2f, "
                  "only complete case analysis is not advised"
                  % metafeatures['RatioOfInstancesWithMissingValues']['value'])
            exploration_results['cca'] = False
        if metafeatures['RatioOfFeaturesWithMissingValues']['value'] > 0.10:
            print("The ratio of features with missing values is not small %.2f, "
                  "only available case analysis is not advised"
                  % metafeatures['RatioOfInstancesWithMissingValues']['value'])
            exploration_results['aca'] = False

        if focus:
            feat = metafeatures['FeaturesWithMostMissingValues']['value']
            ratios = [round(elem, 2) for elem in metafeatures['RatioOfFeaturesWithMostMissingValues']['value']]
            print("The features %s have the most i values, with ratios of respectively %s" % (feat, ratios))


    exploration_results['imbalance'] = False

    # Output imbalance
    if metafeatures['MinClassProbability']['value'] + 0.10 / metafeatures['NumberOfClasses']['value'] \
            < metafeatures['MaxClassProbability']['value']:
        print("\n \033[94m Output imbalance \033[0m")
        exploration_results['imbalance'] = True
        print(
            "Minority class probability is %.2f smaller than the majority class probability. Imbalance seems present,"
            " and should be taken into account" %
            (metafeatures['MaxClassProbability']['value'] - metafeatures['MinClassProbability']['value']))

        if focus:
            num_class = min(3, int(metafeatures['NumberOfClasses']['value'] / 2))
            out = [round(elem, 2) for elem in metafeatures['OutlierClassProbabilities']['value']]
            print("The minority classes %s have a probability of %s" %
                  (metafeatures['OutlierClasses']['value'][0: num_class], out[0: num_class]))
            print("The majority classes %s have a probability of %s" %
                  (metafeatures['OutlierClasses']['value'][-num_class:], out[-num_class:]))

    exploration_results['irrelevance'] = 0

    # Irrelevant features
    if metafeatures['MinCardinalityOfCategoricalFeatures']['value'] == 1 or \
                    metafeatures['MinCardinalityOfNumericFeatures']['value'] == 1:
        print("\n \033[94m Irrelevant Features \033[0m")
        exploration_results['irrelevance'] = metafeatures['MinimumCardinalityFeaturesCount']['value']
        print("%i features without any information are present" % metafeatures['MinimumCardinalityFeaturesCount'][
            'value'])
        exploration_results['irrelevant_features'] = metafeatures['MinimumCardinalityFeatures']['value']


        if focus:
            print('The features without any information present are %s' %
                  list(dfX)[metafeatures['MinimumCardinalityFeatures']['value']].tolist())

    # Normalisation
    print("\n \033[94m Normalisation \033[m")

    exploration_results['norm_means'] = False
    exploration_results['norm_stdev'] = False
    exploration_results['stand'] = False

    mean_means = metafeatures['MeanMeansOfNumericFeatures']['value']
    stdev_means = metafeatures['StdevMeansOfNumericFeatures']['value']
    mean_stdevs = metafeatures['MeanStdDevOfNumericFeatures']['value']
    stdev_stdevs = metafeatures['StdevStdDevOfNumericFeatures']['value']
    if mean_means != 0 and stdev_means / mean_means > 0.01:
        print("There is a high variance in the means of the features. Normalisation or standardisation of the "
              "mean is recommended.")
        exploration_results['norm_means'] = True

    if mean_stdevs != 0 and stdev_stdevs / mean_stdevs > 0.01:
        print("There is a high variance in the standard deviation in the features. Normalisation or standardisation "
              "of the standard deviation is recommended")
        exploration_results['norm_stdev'] = True

    if abs(metafeatures['MeanSkewnessOfNumericFeatures']['value']) > 1:
        print("The average skewness of the numeric features is highly different from the normal distribution. "
              "Keep in mind a significant number of the features is not normally distributed")
        exploration_results['stand'] = True

    if abs(metafeatures['MeanKurtosisOfNumericFeatures']['value'] - 3) > 1:
        print("The average kurtosis of the numeric features is highly different from the normal distribution. "
              "Keep in mind a significant number of the features is not normally distributed")
        exploration_results['stand'] = True

    # Multicollinearity
    print("\n \033[94m Multicollinearity \033[0m")
    mc_threshold = 1 / ((metafeatures['NumberOfFeatures']['value']) ** (1 / 2))

    exploration_results['mc'] = False

    if metafeatures['PredPCA1']['value'] > mc_threshold:
        print('The first principal component already explains %.2f variance in the dataset, showing a high amount of '
              'multicollinearity being present' % metafeatures['PredPCA1']['value'])
        exploration_results['mc'] = True
    elif metafeatures['PredPCA1']['value'] + metafeatures['PredPCA2']['value'] > mc_threshold:
        print(
            'The first two principal component already explain %.2f variance in the dataset, showing a high amount of '
            'multicollinearity being present' %
            metafeatures['PredPCA1']['value'] + metafeatures['PredPCA2']['value'])
        exploration_results['mc'] = True
    elif metafeatures['PredPCA1']['value'] + metafeatures['PredPCA2']['value'] + metafeatures['PredPCA3'][
        'value'] > mc_threshold:
        print(
            'The first three principal component already contain %.2f variance in the dataset, showing a high amount of '
            'multicollinearity being present' %
            metafeatures['PredPCA1']['value'] + metafeatures['PredPCA2']['value'] + metafeatures['PredPCA3']['value'])
        exploration_results['mc'] = True

    if metafeatures['MeanCorrelation']['value'] != 'Too many features for correlation testing':
        if metafeatures['MaxCorrelation']['value'] > 0.5:
            print('At least two features have a high correlation value of %.2f' %
                  metafeatures['MaxCorrelation']['value'])
            exploration_results['mc'] = True
        if metafeatures['MinCorrelation']['value'] < -0.5:
            print('At least two features have a high negative correlation value of %.2f' %
                  metafeatures['MinCorrelation']['value'])
            exploration_results['mc'] = True

        if focus:
            feat = metafeatures['CorrelationOutlierFeatures']['value']
            out = [round(elem, 2) for elem in metafeatures['CorrelationOutliers']['value']]
            print("The combination of the features %s have a negative correlation of %s" % (feat[0:3], out[0:3]))
            print("The combination of the features %s have a positive correlation of %s" % (feat[3:6], out[3:6]))

    if plots:
        plt.show()

    return exploration_results


def _create_pandas_dataframe_dataset(X, y, features, classification=True):
    """ Make a pandas dataframe out of the numpy dataset

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param y: A numpy array of the output. The length of the array should correspond to the size of the first 
    axis of X
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the 
    second axis of X
    :param classification: If the output is categorical. Default: True
    :return: dataframe dfX with data and series dfy with output
    """

    # Create dataframe
    dfX = pd.DataFrame(X, columns=features)

    if classification:
        dfy = pd.Series(y, dtype='category')
    else:
        dfy = pd.Series(y, dtype=float)

    dfy.name = "Output"

    return dfX, dfy


def _prepare_dataset(dfX, missing_values='Unknown'):
    """ Prepare the dataset by changing the missing values and data types to the correct type

    :param dfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :param missing_values: The way missing values are added to the data. Can be a list if missing values are portrayed 
    in multiple ways. Default: 'Unknown' (several well-known ways of portraying missing values are checked)
    :return: dataframe dfX with data, with a correctly matching missing values and data types
    """

    dfX = _prepare_missing_values(dfX, missing_values)

    dfX = _prepare_data_types(dfX)

    return dfX


def _prepare_missing_values(dfX, missing_values='Unknown'):
    """ Prepare the dataset by changing the missing values

    :param dfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :param missing_values: The way missing values are added to the data. Can be a list if missing values are portrayed 
    in multiple ways. Default: 'Unknown' (several well-known ways of portraying missing values are checked)
    :return: dataframe dfX with data, with a correctly matching missing values
    """

    # Change possible missing values to NaN values
    if type(missing_values) == list:
        for missing_value in missing_values:
            dfX = dfX.replace(missing_value, np.NaN)

    elif missing_values == 'Unknown':
        for missing_value in ['', '?', 'None']:
            dfX = dfX.replace(missing_value, np.NaN)

    else:
        dfX = dfX.replace(missing_values, np.NaN)

    return dfX


def _prepare_data_types(dfX):
    """ Prepare the dataset by changing the data types to the correct type

    :param dfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :return: dataframe dfX with data, with a correctly matching data types
    """

    for header in list(dfX):
        X_f = dfX[header].values
        try:
            X_f.astype(float)
            dfX[header] = pd.to_numeric(dfX[header], errors='coerce')
        except:
            dfX[header] = dfX[header].astype('category')

    return dfX




