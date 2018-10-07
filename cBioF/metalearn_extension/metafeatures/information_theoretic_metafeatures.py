import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from .common_operations import *

def get_entropy(col):
    return entropy(col.value_counts())

def get_class_entropy(Y_sample):
    return (get_entropy(Y_sample),)

def get_attribute_entropy(feature_array):
    entropies = [get_entropy(feature) for feature in feature_array]
    mean_attribute_entropy, _, min_attribute_entropy, quartile1_attribute_entropy, quartile2_attribute_entropy, quartile3_attribute_entropy, max_attribute_entropy = profile_distribution(entropies)
    return (mean_attribute_entropy, min_attribute_entropy, quartile1_attribute_entropy, quartile2_attribute_entropy, quartile3_attribute_entropy, max_attribute_entropy)

# START ADDITIONAL METAFEATURES code
def get_outliers_attribute_entropy(feature_array):
    entropies = [get_entropy(feature) for  feature in feature_array]

    attribute_entropy_outliers, indices_attribute_entropy_outlier_features = return_most_important(entropies)

    attribute_entropy_outlier_features = [list(feature_array)[i].name for i in indices_attribute_entropy_outlier_features]

    return attribute_entropy_outlier_features, attribute_entropy_outliers

def get_histograms_outliers_attribute_entropy(feature_array):
    # entropies = [get_entropy(feature) for feature in feature_array]
    entropies = [get_entropy(feature_array[i]) for i in range(len(feature_array))]

    attribute_entropy_outliers, indices_attribute_entropy_outlier_features = return_most_important(entropies)

    attribute_entropy_outlier_features = [list(feature_array)[i].name for i in
                                          indices_attribute_entropy_outlier_features]

    #outlier_arrays = [feature_array[i] for i in indices_attribute_entropy_outlier_features]
    outlier_arrays = [feature_array[indices_attribute_entropy_outlier_features[i]]
                      for i in range(len(indices_attribute_entropy_outlier_features))]

    outlier_titles = ["%s: %.2f" % (attribute_entropy_outlier_features[i], attribute_entropy_outliers[i])
                      for i in range(len(attribute_entropy_outliers))]

    get_outlier_histograms(outlier_arrays, "Attribute entropy outliers", outlier_titles)
    return ['No values for plots']

def get_boxplots_outliers_attribute_entropy(feature_array):
    entropies = [get_entropy(feature) for feature in feature_array]

    attribute_entropy_outliers, indices_attribute_entropy_outlier_features = return_most_important(entropies)

    attribute_entropy_outlier_features = [list(feature_array)[i].name for i in
                                          indices_attribute_entropy_outlier_features]

    outlier_arrays = [feature_array[i] for i in indices_attribute_entropy_outlier_features]

    outlier_titles = ["%s: %.2f" % (attribute_entropy_outlier_features[i], attribute_entropy_outliers[i])
                      for i in range(len(attribute_entropy_outliers))]

    get_outlier_boxplots(outlier_arrays, "Attribute entropy outliers", outlier_titles)
    return ['No values for plots']
# END ADDITIONAL METAFEATURES code

def get_joint_entropy(feature_class_array):
    entropies = [get_entropy(feature_class_pair[0].astype(str) + feature_class_pair[1].astype(str)) for feature_class_pair in feature_class_array]
    mean_joint_entropy, _, min_joint_entropy, quartile1_joint_entropy, quartile2_joint_entropy, quartile3_joint_entropy, max_joint_entropy = profile_distribution(entropies)
    return (mean_joint_entropy, min_joint_entropy, quartile1_joint_entropy, quartile2_joint_entropy, quartile3_joint_entropy, max_joint_entropy)

# START ADDITIONAL METAFEATURES code
def get_outliers_joint_entropy(feature_class_array):
    entropies = [get_entropy(feature_class_pair[0].astype(str) + feature_class_pair[1].astype(str)) for
                 feature_class_pair in feature_class_array]

    joint_entropy_outliers, indices_joint_entropy_outlier_features = return_most_important(entropies)

    joint_entropy_outlier_features = [feature_class_array[i][0].name for i in indices_joint_entropy_outlier_features]

    return joint_entropy_outlier_features, joint_entropy_outliers

def get_histograms_outliers_joint_entropy(feature_class_array):
    entropies = [get_entropy(feature_class_pair[0].astype(str) + feature_class_pair[1].astype(str)) for
                 feature_class_pair in feature_class_array]
    joint_entropy_outliers, indices_joint_entropy_outlier_features = return_most_important(entropies)
    joint_entropy_outlier_features = [list(feature_class_array)[i][0].name for i in
                                      indices_joint_entropy_outlier_features]

    outlier_arrays = [feature_class_array[i][0].astype(str) for i in indices_joint_entropy_outlier_features]
    outlier_titles = ["%s: %.2f" % (joint_entropy_outlier_features[i], joint_entropy_outliers[i])
                      for i in range(len(joint_entropy_outliers))]

    get_outlier_histograms(outlier_arrays, "Joint entropy outliers", outlier_titles)
    return ['No values for plots']

def get_mutual_information(feature_class_array):
    mi_scores = [mutual_info_score(*feature_class_pair) for feature_class_pair in feature_class_array]
    mean_mutual_information, _, min_mutual_information, quartile1_mutual_information, quartile2_mutual_information, quartile3_mutual_information, max_mutual_information = profile_distribution(mi_scores)
    return (mean_mutual_information, min_mutual_information, quartile1_mutual_information, quartile2_mutual_information, quartile3_mutual_information, max_mutual_information)

# START ADDITIONAL METAFEATURES code
def get_outliers_mutual_information(feature_class_array):
    mi_scores = [mutual_info_score(*feature_class_pair) for feature_class_pair in feature_class_array]

    mi_outliers, indices_mi_outlier_features = return_most_important(mi_scores)

    mi_outlier_features = [feature_class_array[i][0].name for i in indices_mi_outlier_features]

    return mi_outlier_features, mi_outliers

def get_histograms_outliers_mutual_information(feature_class_array):
    mi_scores = [mutual_info_score(*feature_class_pair) for feature_class_pair in feature_class_array]

    mutual_information_outliers, indices_mutual_information_outlier_features = return_most_important(mi_scores)

    mutual_information_outlier_features = [feature_class_array[i][0].name for i in indices_mutual_information_outlier_features]


    outlier_arrays = [feature_class_array[i][0].astype(str) for i in indices_mutual_information_outlier_features]

    outlier_titles = ["%s: %.2f" % (mutual_information_outlier_features[i], mutual_information_outliers[i])
                      for i in range(len(mutual_information_outliers))]

    get_outlier_histograms(outlier_arrays, "Mutual information outliers", outlier_titles)
    return ['No values for plots']
# END ADDITIONAL METAFEATURES code

def get_equivalent_number_features(class_entropy, mutual_information):
    if mutual_information == 0:
        enf = np.nan
    else:
        enf = class_entropy / mutual_information
    return (enf,)

def get_noise_signal_ratio(attribute_entropy, mutual_information):
    if mutual_information == 0:
        nsr = np.nan
    else:
        nsr = (attribute_entropy - mutual_information) / mutual_information
    return (nsr,)
