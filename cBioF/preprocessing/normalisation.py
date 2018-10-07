# -*- coding: utf-8 -*-
"""Generic normalising the numeric features"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>


from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler as SS

from cBioF.preprocessing.find_data_types import column_types_dataset


def normalise_numeric_features(X, standardisation=False, means=True, stdev=True):
    """ Normalisation for numeric features

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param standardisation: Whether standardisation needs to be done instead of normalisation. Default: False
    :param means: Whether the mean should be normalised. Default: True
    :param stdev: Whether the standard devation should be normalised. Default: True
    :return: X and features with numeric features normalised.
    """

    column_types = column_types_dataset(X, categorical=False)

    for i in range(len(column_types)):
        if column_types[i]:

            if standardisation:
                # Standardisation
                scaler = MMS([0, 1])
                X[:, i:i+1] = scaler.fit_transform(X[:, i:i+1])
            else:
                # Normalisation
                scaler = SS(means, stdev)
                X[:, i:i+1] = scaler.fit_transform(X[:, i:i+1])

    return X