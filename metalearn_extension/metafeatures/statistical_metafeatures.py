import time
import math
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

from .common_operations import *

# START ADDITIONAL EXPLORATION imports
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
# END ADDITIONAL EXPLORATION imports

warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning) # suppress sklearn warnings

def get_numeric_means(numeric_features_array):
    means = [feature.mean() for feature in numeric_features_array]
    return profile_distribution(means)

def get_numeric_stdev(numeric_features_array):
    stdevs = [feature.std() for feature in numeric_features_array]
    return profile_distribution(stdevs)

def get_numeric_skewness(numeric_features_array):
    skews = [feature.skew() for feature in numeric_features_array]
    return profile_distribution(skews)

def get_numeric_kurtosis(numeric_features_array):
    kurtoses = [feature.kurtosis() for feature in numeric_features_array]
    return profile_distribution(kurtoses)

# START ADDITIONAL EXPLORATION code
def get_distribution_boxplots(numeric_features_array):
    means = [feature.mean() for feature in numeric_features_array]
    stdevs = [feature.std() for feature in numeric_features_array]
    skews = [feature.skew() for feature in numeric_features_array]
    kurtoses = [feature.kurtosis() for feature in numeric_features_array]

    fig, axes = plt.subplots(2, 2)

    fig.suptitle("Distributions of numerical features")

    axes[0, 0].boxplot(means)
    axes[0, 0].set_title("Means")
    axes[0, 0].set_ylabel("Values", verticalalignment='bottom')

    axes[0, 1].boxplot(stdevs)
    axes[0, 1].set_title("Standard deviations")
    axes[0, 1].set_ylabel("Values", verticalalignment='bottom')

    axes[1, 0].boxplot(skews)
    axes[1, 0].set_title("Skewnesses")
    axes[1, 0].set_ylabel("Values", verticalalignment='bottom')

    axes[1, 1].boxplot(kurtoses)
    axes[1, 1].set_title("Kurtoses")
    axes[1, 1].set_ylabel("Values", verticalalignment='bottom')

    plt.draw()

    return ['No value for plots']
# END ADDITIONAL EXPLORATION code

def get_pca(X_preprocessed):
    num_components = min(3, X_preprocessed.shape[1])
    pca_data = PCA(n_components=num_components)
    pca_data.fit_transform(X_preprocessed.values)
    pred_pca = pca_data.explained_variance_ratio_
    pred_eigen = pca_data.explained_variance_
    pred_det = np.linalg.det(pca_data.get_covariance())
    variance_percentages = [0] * 3
    for i in range(len(pred_pca)):
        variance_percentages[i] = pred_pca[i]
    eigenvalues = [0] * 3
    for i in range(len(pred_eigen)):
        eigenvalues[i] = pred_eigen[i]
    return (variance_percentages[0], variance_percentages[1], variance_percentages[2], eigenvalues[0], eigenvalues[1], eigenvalues[2], pred_det)

def get_correlations(X_sample, column_types):

    # START ADDITIONAL EXPLORATION code
    if X_sample.shape[1] > 1000:
        return 'Too many features for correlation testing', 'Too many features for correlation testing', 'Too many features for correlation testing', 'Too many features for correlation testing'

    correlations = get_canonical_correlations(X_sample, column_types)
    mean_correlation, stdev_correlation, min_correlation, _, _, _, max_correlation = profile_distribution(correlations)
    return (mean_correlation, stdev_correlation, min_correlation, max_correlation)
    # END ADDITIONAL EXPLORATION code

# START ADDITIONAL EXPLORATION code
def get_outlier_correlations(X_sample, column_types):

    if X_sample.shape[1] > 1000:
        return 'Too many features for correlation testing', 'Too many features for correlation testing'

    correlations = get_canonical_correlations(X_sample, column_types)
    correlation_outliers, indices_correlation_outlier_features = \
        return_most_important(correlations, return_end="Both")

    correlation_possibilities = []

    for col_name_i, col_name_j in itertools.combinations(list(X_sample), 2):
        correlation_possibilities.append([col_name_i, col_name_j])

    correlation_outlier_features = [correlation_possibilities[i] for i in indices_correlation_outlier_features]

    return correlation_outlier_features, correlation_outliers
# END ADDITIONAL EXPLORATION code

def get_correlations_by_class(X_sample, Y_sample):
    correlations = []
    XY = pd.concat([X_sample,Y_sample], axis=1)
    XY_grouped_by_class = XY.groupby(Y_sample.name)
    for label in Y_sample.unique():
        group = XY_grouped_by_class.get_group(label).drop(Y_sample.name, axis=1)
        correlations.extend(get_canonical_correlations(group))
    mean_correlation, stdev_correlation, _, _, _, _, _ = profile_distribution(correlations)
    return (mean_correlation, stdev_correlation)

def get_canonical_correlations(dataframe, column_types):
    '''
    computes the correlation coefficient between each distinct pairing of columns
    preprocessing note:
        any rows with missing values (in either paired column) are dropped for that pairing
        categorical columns are replaced with one-hot encoded columns
        any columns which have only one distinct value (after dropping missing values) are skipped
    returns a list of the pairwise canonical correlation coefficients
    '''

    def preprocess(series):
        if column_types[series.name] == 'CATEGORICAL':
            series = pd.get_dummies(series)
        array = series.values.reshape(series.shape[0], -1)
        return array

    if dataframe.shape[1] < 2:
        return []

    correlations = []
    skip_cols = set()
    for col_name_i, col_name_j in itertools.combinations(dataframe.columns, 2):
        if col_name_i in skip_cols or col_name_j in skip_cols:
            correlations.append(0)
            continue

        df_ij = dataframe[[col_name_i, col_name_j]].dropna(axis=0, how="any")
        col_i = df_ij[col_name_i]
        col_j = df_ij[col_name_j]

        if np.unique(col_i).shape[0] <= 1:
            skip_cols.add(col_name_i)
            correlations.append(0)
            continue
        if np.unique(col_j).shape[0] <= 1:
            skip_cols.add(col_name_j)
            correlations.append(0)
            continue

        col_i = preprocess(col_i)
        col_j = preprocess(col_j)

        col_i_c, col_j_c = CCA(n_components=1).fit_transform(col_i,col_j)

        if np.unique(col_i_c).shape[0] <= 1 or np.unique(col_j_c).shape[0] <= 1:
            c = 0
        else:
            c = np.corrcoef(col_i_c.T, col_j_c.T)[0,1]
        correlations.append(c)

    return correlations
