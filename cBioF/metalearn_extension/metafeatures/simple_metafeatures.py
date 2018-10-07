import numpy as np
from pandas import DataFrame

from .common_operations import *

def get_dataset_stats(X, column_types):
    number_of_instances = X.shape[0]
    number_of_features = X.shape[1]
    numeric_features = len(get_numeric_features(X, column_types))
    categorical_features = number_of_features - numeric_features
    ratio_of_numeric_features = numeric_features / number_of_features
    ratio_of_categorical_features = categorical_features / number_of_features
    return (number_of_instances, number_of_features, numeric_features, categorical_features, ratio_of_numeric_features, ratio_of_categorical_features)

def get_dimensionality(number_of_features, number_of_instances):
    dimensionality = number_of_features / number_of_instances
    return (dimensionality,)

def get_missing_values(X):
    missing_values_by_instance = X.shape[1] - X.count(axis=1)
    missing_values_by_feature = X.shape[0] - X.count(axis=0)
    number_missing = int(np.sum(missing_values_by_instance)) # int for json compatibility
    ratio_missing = number_missing / (X.shape[0] * X.shape[1])
    number_instances_with_missing = int(np.sum(missing_values_by_instance != 0)) # int for json compatibility
    ratio_instances_with_missing = number_instances_with_missing / X.shape[0]
    number_features_with_missing = int(np.sum(missing_values_by_feature != 0))
    ratio_features_with_missing = number_features_with_missing / X.shape[1]
    return (
        number_missing, ratio_missing, number_instances_with_missing,
        ratio_instances_with_missing, number_features_with_missing,
        ratio_features_with_missing
    )

# START ADDITIONAL EXPLORATION code
def get_most_missing_values(X):

    ratio_missing_values_by_instance = (X.shape[1] - X.count(axis=1)) / X.shape[1]
    ratio_missing_values_by_feature = (X.shape[0] - X.count(axis=0)) / X.shape[0]

    ratio_most_missing_values_by_instance, most_missing_values_by_instance = \
        return_most_important(ratio_missing_values_by_instance, return_end='Top')
    ratio_most_missing_values_by_feature, indices_most_missing_values_by_feature = \
        return_most_important(ratio_missing_values_by_feature, return_end='Top')

    most_missing_values_by_feature = [list(X)[i] for i in indices_most_missing_values_by_feature]

    return(most_missing_values_by_instance, ratio_most_missing_values_by_instance,
           most_missing_values_by_feature, ratio_most_missing_values_by_feature)
# END ADDITIONAL EXPLORATION code

def get_class_stats(Y):
    classes = Y.unique()
    number_of_classes = classes.shape[0]
    counts = [sum(Y == label) for label in classes]
    probs = [count/Y.shape[0] for count in counts]
    mean_class_probability, stdev_class_probability, min_class_probability, _, _, _, max_class_probability = profile_distribution(probs)
    majority_class_size = max(counts)
    minority_class_size = min(counts)
    return (number_of_classes, mean_class_probability, stdev_class_probability, min_class_probability, max_class_probability, minority_class_size, majority_class_size)

# START ADDITIONAL EXPLORATION code
def get_outlier_class_stats(Y):
    classes = Y.unique()
    counts = [sum(Y == label) for label in classes]
    probs = [count / Y.shape[0] for count in counts]

    probs_outlier_classes, indices_outlier_classes = return_most_important(probs, return_end='Both')
    outlier_classes = [classes[i] for i in indices_outlier_classes]

    return(outlier_classes, probs_outlier_classes)

def get_histogram_class_stats(Y):
    classes = Y.unique()
    counts = [sum(Y == label) for label in classes]

    plt.bar(classes.astype(str), counts)
    plt.title("The output class distribution")
    plt.xlabel("Output classes")
    plt.ylabel("Number of instances")

    plt.draw()

    return ['No output for plots']
# END ADDITIONAL EXPLORATION code

def get_categorical_cardinalities(X, column_types):
    cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_categorical_features(X, column_types)]
    mean_cardinality_of_categorical_features, stdev_cardinality_of_categorical_features, min_cardinality_of_categorical_features, _, _, _, max_cardinality_of_categorical_features = profile_distribution(cardinalities)
    return (mean_cardinality_of_categorical_features, stdev_cardinality_of_categorical_features, min_cardinality_of_categorical_features, max_cardinality_of_categorical_features)

# START ADDITIONAL EXPLORATION code
def get_outlier_categorical_cardinalities(X, column_types):
    cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_categorical_features(X, column_types)]

    categorical_feature_names = get_location_categorical_features(X, column_types)

    categorical_cardinality_outliers, indices_categorical_cardinality_outlier_features = \
        return_most_important(cardinalities, return_end="Both")
    categorical_cardinality_outlier_features = [list(X)[categorical_feature_names[i]] for i in indices_categorical_cardinality_outlier_features]

    return categorical_cardinality_outlier_features, categorical_cardinality_outliers
# END ADDITIONAL EXPLORATION code

def get_numeric_cardinalities(X, column_types):
    cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_numeric_features(X, column_types)]
    mean_cardinality_of_numeric_features, stdev_cardinality_of_numeric_features, min_cardinality_of_numeric_features, _, _, _, max_cardinality_of_numeric_features = profile_distribution(cardinalities)
    return (mean_cardinality_of_numeric_features, stdev_cardinality_of_numeric_features, min_cardinality_of_numeric_features, max_cardinality_of_numeric_features)

# START ADDITIONAL EXPLORATION code
def get_outlier_numeric_cardinalities(X, column_types):
    cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_numeric_features(X, column_types)]

    numeric_feature_names = get_location_numeric_features(X, column_types)

    numeric_cardinality_outliers, indices_numeric_cardinality_outlier_features = \
        return_most_important(cardinalities, return_end="Both")
    numeric_cardinality_outlier_features = [list(X)[numeric_feature_names[i]] for i in indices_numeric_cardinality_outlier_features]

    return numeric_cardinality_outlier_features, numeric_cardinality_outliers

def get_minimum_cardinality_number(X, column_types):
    categorical_cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_categorical_features(X, column_types)]
    numeric_cardinalities = [X[feature].unique().shape[0] for feature in get_numeric_features(X, column_types)]

    categorical_feature_names = get_location_categorical_features(X, column_types)
    numeric_feature_names = get_location_numeric_features(X, column_types)

    if len(categorical_cardinalities) == 0:
        categorical_cardinalities = [float("inf")]
    elif len(numeric_cardinalities) == 0:
        numeric_cardinalities == [float("inf")]

    min_cardinality = min(min(categorical_cardinalities), min(numeric_cardinalities))

    min_feature_names = []
    loc_min_categorical_cardinality = np.argwhere(np.asarray(categorical_cardinalities) == min_cardinality).flatten()
    loc_min_numeric_cardinality = np.argwhere(np.asarray(numeric_cardinalities) == min_cardinality).flatten()

    min_feature_names = [categorical_feature_names[i] for i in loc_min_categorical_cardinality.tolist()]
    min_feature_names.extend([numeric_feature_names[i] for i in loc_min_numeric_cardinality.tolist()])

    return len(min_feature_names), min_feature_names

def get_histograms_outliers_categorical_cardinalities(X, column_types):
    cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_categorical_features(X, column_types)]

    categorical_feature_names = get_location_categorical_features(X, column_types)

    categorical_cardinality_outliers, indices_categorical_cardinality_outlier_features = \
        return_most_important(cardinalities, return_end="Both")
    categorical_cardinality_outlier_features = [list(X)[categorical_feature_names[i]] for i in
                                            indices_categorical_cardinality_outlier_features]

    outlier_arrays = [X[feature].dropna() for feature in categorical_cardinality_outlier_features]

    outlier_titles = ["%s: %.2f" % (categorical_cardinality_outlier_features[i], categorical_cardinality_outliers[i])
                      for i in range(len(categorical_cardinality_outliers))]

    get_outlier_histograms(outlier_arrays, "Categorical cardinality outliers", outlier_titles)

    return ['No output for plots']

def get_boxplots_outliers_numeric_cardinalities(X, column_types):
    cardinalities = [X[feature].dropna().unique().shape[0] for feature in get_numeric_features(X, column_types)]

    # features = get_numeric_features(X, column_types)
    # cardinalities = [X[features[i]].dropna().unique().shape[0] for i in range(len(features))]

    numeric_feature_names = get_location_numeric_features(X, column_types)

    numeric_cardinality_outliers, indices_numeric_cardinality_outlier_features = \
        return_most_important(cardinalities, return_end="Both")
    numeric_cardinality_outlier_features = [list(X)[numeric_feature_names[i]] for i in
                                            indices_numeric_cardinality_outlier_features]

    outlier_arrays = [X[feature].dropna() for feature in numeric_cardinality_outlier_features]
    # outlier_arrays = [X[features[i]].dropna() for i in range(len(features))]

    outlier_titles = ["%s: %.2f" % (numeric_cardinality_outlier_features[i], numeric_cardinality_outliers[i])
                      for i in range(len(numeric_cardinality_outliers))]

    get_outlier_boxplots(outlier_arrays, "Numeric cardinality outliers", outlier_titles)

    return ['No output for plots']
# END ADDITIONAL EXPLORATION code