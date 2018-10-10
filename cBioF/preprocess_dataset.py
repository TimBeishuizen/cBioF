import numpy as np
import pandas as pd
from cBioF.feature_selection import wrapper_methods as WM, order_methods as OM
from cBioF.missing_value_handling import list_deletion as LDM, value_imputation as impute
from cBioF.preprocessing import hot_encoding as HE, normalisation as NS
from cBioF._robustness_methods import robustness_methods
from sklearn.feature_selection import SelectFwe as SF, f_classif

from cBioF.missing_value_handling import impute


def pandas_preprocess_dataset(dfX, dfy, exploration_results, fs_example=False):
    """ Preprocess the data according to earlier performed exploration results with found issues. These issues are based on:
     - feature types,
     - feature dimensionality,
     - missing values,
     - output imbalance,
     - irrelevant features,
     - normalisation,
     - multicollinearity

    Since feature selection can be very dataset specific, it can also be removed from the preprocessing list.

    :param dfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :param dfy: A pandas series of the output. The length of the array should correspond to the size of the first
    axis of X
    :param exploration_results: A dict with the results of the earlier exploration, corresponding to the aforementioned
    issues
    :param fs_example: Whether also an example of feature selection should be done. Default: False
    :return: The preprocessed dfX and dfy
    """

    # Test if input is according to standards
    robustness_methods.check_pandas_input(dfX, dfy)

    dfX = dfX.replace(np.NaN, '')

    X = dfX.values
    y = dfy.cat.codes.values
    features = np.asarray(list(dfX))

    X_new, y_new, f_new = preprocess_dataset(X, y, features, exploration_results, fs_example)

    dfX_new = pd.DataFrame(X_new, columns=f_new)
    dfy_new = pd.Series(y_new)

    return dfX_new, dfy_new


def preprocess_dataset(X, y, features, exploration_results, fs_example=False):
    """ Preprocess the data according to earlier performed exploration results with found issues. These issues are based on:
     - feature types,
     - feature dimensionality,
     - missing values,
     - output imbalance,
     - irrelevant features,
     - normalisation,
     - multicollinearity

    Since feature selection can be very dataset specific, it can also be removed from the preprocessing list.

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param y: A numpy array of the output. The length of the array should correspond to the size of the first
    axis of X
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the
    second axis of X
    :param exploration_results: A dict with the results of the earlier exploration, corresponding to the aforementioned
    issues
    :param fs_example: Whether also an example of feature selection should be done. Default: False
    :return: The preprocessed X, y and features
    """

    # Test the input to be according to the standards
    robustness_methods.check_input_arrays(X, y, features)

    # First change data for missing values
    if exploration_results['mv']:
        print("\nStarting missing value handling...")
        old_features = np.copy(features)
        if exploration_results['cca']:
            X, y = LDM.cca(X, y, missing_values='')
        elif exploration_results['aca']:
            X, features = LDM.aca(X, features, missing_values='')
        else:
            X, features = LDM.aca(X, features, missing_values='', removal_fraction=0.15)

            X = impute.mean_imputation(X, missing_values='')

        removed_features = _return_removed_features(features, old_features)

        print("These features are removed due to having too many missing values: %s" % removed_features)

    if exploration_results['irrelevance'] > 0:
        print("\nRemoving irrelevant features...")
        # Remove irrelevant
        irr_feat_loc = exploration_results['irrelevant_features']
        X = np.delete(X, irr_feat_loc, axis=1)
        old_features = np.copy(features)
        features = np.delete(features, irr_feat_loc)
        removed_features = _return_removed_features(features, old_features)

        print("These features are removed due to having no information: %s" % removed_features)

        _return_removed_features(features, old_features)

    if exploration_results['norm_means'] or exploration_results['norm_stdev']:
        print("\nNormalising numeric features...")
        # Normalise or standardise values
        NS.normalise_numeric_features(X, exploration_results['stand'],
                                   exploration_results['norm_means'], exploration_results['norm_stdev'])

    # Than change categorical to numeric values
    if exploration_results['cat']:
        print("\nHot encoding categorical values...")
        X, features = HE.hot_encode_categorical_features(X, features)

    if exploration_results['fs'] and fs_example:
        print("\nDoing an example of feature selection...")
        # Feature selection if multicollinearity
        if exploration_results['mc'] and False:
            # Remove multicollinearity
            feature_selector = WM.ForwardSelector(threshold=0.001)

            # Order to have more relevant features first
            feature_orderer = OM.FeatureOrderer(f_classif)
            X = feature_orderer.fit_transform(X, y)
            features = features[np.argsort(-feature_orderer.scores_)]
        else:
            feature_selector = SF(f_classif, alpha=0.05)

        # Transform data to feature_selection
        X = feature_selector.fit_transform(X, y)
        old_features = np.copy(features)
        features = features[feature_selector.get_support()]
        removed_features = _return_removed_features(features, old_features)

        print("These features are removed due to feature selection: %s" % removed_features)

    # if exploration_results['imbalance']:
    #     print("Try to use F1-score over Accuracy in quality measurements")
    #     # F1 score

    return X, y, features


def _return_removed_features(new_features, old_features):
    """ A simple method that shows the difference between the old feature set and new feature set

    :param new_features: The new feature set
    :param old_features: The old feature set
    :return: The feature that appeared in the old set, but not in the new set.
    """

    # Show removed features
    removed_features = []
    for feat in old_features:
        if feat not in new_features:
            removed_features.append(feat)

    return removed_features