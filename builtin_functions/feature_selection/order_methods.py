# -*- coding: utf-8 -*-
"""Generic feature orderer mixin"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection.base import SelectorMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import StandardScaler

import numpy as np
import copy

class FeatureOrderer(BaseEstimator, TransformerMixin):
    """ An class based on ordering the features in a specific manner for methods that ordering is important for.
    The ordering can be done in two different ways:
    - random, cr4eating a random permutation of the ordered features
    - order_function that creates a value for feature to order it.

    """

    def __init__(self, score_func):

        """ Initialisation of the FeatureOrderer class

        :param score_func: The way of ordering the features
        """

        self.score_func = score_func
        self.order = None

    def fit(self, X, y):
        """ Fit the data of X and y to find the order needed.

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        # Find out if random or not
        if str(self.score_func) == 'random':
            score_func_ret = np.random.permutation(X.shape[1])
        else:
            score_func_ret = self.score_func(X, y)

        # Give scores based to the score function
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        return self

    def transform(self, X, y=None, features=None):
        """ Transform the dataset into the desired order. If feature names are important, they can be ordered as well

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :param features: A numpy array of the feature names. The length of the array should correspond to the size of the
        second axis of X
        :return: Ordered data matrix and possibly ordered feature array
        """

        order = np.argsort(-self.scores_)

        if features is None:
            return X[:, order]
        else:
            return X[:, order], features[order]


