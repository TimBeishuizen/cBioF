import numpy as np
import pandas as pd

# START ADDITIONAL EXPLORATION imports
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def profile_distribution(data):
    """
    Compute the mean, standard deviation, min, quartile1, quartile2, quartile3, and max of a vector

    Parameters
    ----------
    data: array of real values

    Returns
    -------
    features = dictionary containing the min, max, mean, and standard deviation
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        ddof = 1 if len(data) > 1 else 0
        dist_mean = np.mean(data)
        dist_stdev = np.std(data, ddof=ddof)
        dist_min, dist_quartile1, dist_quartile2, dist_quartile3, dist_max = np.percentile(data, [0,25,50,75,100])
    return (dist_mean, dist_stdev, dist_min, dist_quartile1, dist_quartile2, dist_quartile3, dist_max)

def get_numeric_features(dataframe, column_types):
    return [feature for feature in dataframe.columns if column_types[feature] == "NUMERIC"]

def get_categorical_features(dataframe, column_types):
    return [feature for feature in dataframe.columns if column_types[feature] == "CATEGORICAL"]

def dtype_is_numeric(dtype):
    return "int" in str(dtype) or "float" in str(dtype)

# START ADDITIONAL METAFEATURES code
def get_location_categorical_features(dataframe, column_types):
    return [index for index in range(len(dataframe.columns)) if column_types[dataframe.columns[index]] == "CATEGORICAL"]

def get_location_numeric_features(dataframe, column_types):
    return [index for index in range(len(dataframe.columns)) if column_types[dataframe.columns[index]] == "NUMERIC"]


def return_most_important(values, return_end='Both', return_number=3):
    """

    :param values:
    :param return_end:
    :param return_number:
    :return:
    """

    order = np.argsort(values)

    return_values = []
    return_features = []

    sorted_values = [values[i] for i in order]
    sorted_features = order

    if return_end in ['Bottom', 'Both']:
        return_values.extend(sorted_values[:return_number])
        return_features.extend(sorted_features[:return_number])

    if return_end in ['Top', 'Both']:
        return_values.extend(sorted_values[-return_number:])
        return_features.extend(sorted_features[-return_number:])

    return return_values, return_features

def get_outlier_boxplots(feature_values, title, subtitles):
    """

    :param feature_values:
    :param title:
    :param subtitles:
    :return:
    """

    half_way = round(len(feature_values) / 2)

    fig, axes = plt.subplots(2, half_way)

    fig.suptitle(title)

    for i in range(half_way):
        axes[0, i].boxplot(feature_values[i].values)
        axes[0, i].set_title(subtitles[i])
        axes[0, i].set_ylabel('values', verticalalignment='bottom', size='small')

        if i + half_way < len(feature_values):
            axes[1, i].boxplot(feature_values[i + half_way].values)
            axes[1, i].set_title(subtitles[i + half_way])
            axes[1, i].set_ylabel('values', verticalalignment='bottom', size='small')

    plt.draw()

    return ['No output for plots']

def get_outlier_histograms(feature_values, title, subtitles):
    """

    :param feature_values:
    :param title:
    :param subtitles:
    :return:
    """

    half_way = round(len(feature_values) / 2)

    fig, axes = plt.subplots(2, half_way)

    fig.suptitle(title)

    for ax in fig.axes:
        matplotlib.pyplot.sca(ax)
        plt.xticks(rotation=30, ha='right', size='xx-small')
        plt.yticks(size='xx-small')

    for i in range(half_way):

        names = np.unique(feature_values[i])
        values = [np.count_nonzero(feature_values[i] == name) for name in names]

        axes[0, i].bar(names, values)
        axes[0, i].set_title(subtitles[i], verticalalignment='top', size='medium')
        axes[0, i].set_xlabel('categories', verticalalignment='bottom', size='small')
        axes[0, i].set_ylabel('instance count', verticalalignment='bottom', size='small')

        if i + half_way < len(feature_values):
            names = np.unique(feature_values[i + half_way])
            values = [np.count_nonzero(feature_values[i + half_way] == name) for name in names]

            axes[1, i].bar(names, values)
            axes[1, i].set_title(subtitles[i + half_way], verticalalignment='top', size='medium')
            axes[1, i].set_xlabel('categories', verticalalignment='bottom', size='small')
            axes[1, i].set_ylabel('instance count', verticalalignment='bottom', size='small')

    plt.draw()

    return ['No output for plots']
# END ADDITIONAL METAFEATURES code