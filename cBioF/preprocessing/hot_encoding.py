# -*- coding: utf-8 -*-
"""Generic hot encoding the categorical features"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>


import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer as MLB

from cBioF.preprocessing.find_data_types import column_types_dataset


def hot_encode_categorical_features(X, features):
    """ Hot encoding for categorical features

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the
    second axis of X
    :return: X and features with categorical features hot encoded
    """

    column_types = column_types_dataset(X)

    X_new = np.zeros((X.shape[0], 0))
    f_new = []

    for i in range(len(column_types)):
        if column_types[i]:
            # Hot encode categories
            hot_encoder = MLB()
            new_col = hot_encoder.fit_transform(X[:, i:i+1])

            X_new = np.append(X_new, new_col, axis=1)
            for label in hot_encoder.classes_:
                f_new.append(features[i] + '_' + label)
        else:
            X_new = np.append(X_new, X[:, i:i+1], axis=1)
            f_new.append(features[i])

    return X_new, np.asarray(f_new)