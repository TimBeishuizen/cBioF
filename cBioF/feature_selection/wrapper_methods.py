# -*- coding: utf-8 -*-
"""Generic feature selection mixin"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>

from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection.base import SelectorMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

import numpy as np
import copy


class WrapperSelector(BaseEstimator, SelectorMixin):
    """ A base class for the feature selection wrapper methods. It initialized a way to find whihch features are selected
    as well as a score computation to be used for the wrapper methods.

    """

    def __init__(self, threshold=0.01, score_func=GaussianNB(), cv_groups=5):
        """ Initialisation of the wrapper selector, defining several important variables beforehand

        :param threshold: The threshold for change in quality with new features
        :param score_func: The score function that computes the quality of the change
        :param cv_groups: The number of cross validation groups to be used with the score function
        """

        self.score_func = score_func
        self.threshold = threshold
        self.cv_groups = cv_groups
        self.selected_features = None
        self.score = 0

    def _get_support_mask(self):
        """ A mask that shows which features are selected, being an array with booleans

        :return: The mask with feature selections
        """

        if self.selected_features is None:
            raise ValueError("First fit the model before transform")

        return self.selected_features

    def _compute_score(self, X, y):
        """ Compute the score for the new candidate feature set for possible addition or removal of a newly proposed feature

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: The score
        """

        if X.shape[0] == 0:
            return 0
        else:
            return np.mean(cross_val_score(estimator=self.score_func, X=X, y=y, cv=self.cv_groups))


class ForwardSelector(WrapperSelector):
    """ A feature selection wrapper method, based on forward sequential selection

    """

    def fit(self, X, y):
        """ Fit the data of X and y to find the features to be selected

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        # Loop through all features to find the best
        self.selected_features = np.full(X.shape[1], False, dtype=bool)

        self._forward_selection(X, y)

        return self

    def _forward_selection(self, X, y):
        """ Use forward selection to find the features to be selected

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        for i in range(X.shape[1]):

            # If feature already used (useful for further implementations)
            if self.selected_features[i]:
                continue

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = True

            # Compute the new score with the added feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Add feature if chance is big enough
            if new_score - self.threshold > self.score:
                self.selected_features[i] = True
                self.score = new_score

        return self


class BackwardSelector(WrapperSelector):
    """ A feature selection wrapper method, based on backward sequential selection

    """

    def fit(self, X, y):
        """ Fit the data of X and y to find the features to be selected.

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        # Loop through all features to remove the worst
        self.selected_features = np.full(X.shape[1], True, dtype=bool)

        self._backward_selection(X, y)

        return self

    def _backward_selection(self, X, y):
        """ Use backward selection to find the features to be selected

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        # Loop through processed features to remove the worst
        for i in reversed(range(X.shape[1])):

            # If already false (useful for further implementations)
            if not self.selected_features[i]:
                continue

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = False

            # Compute the new score with the removed feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Remove feature if chance is big enough
            if new_score + self.threshold > self.score:
                self.selected_features[i] = False
                self.score = new_score

        return


class PTA(WrapperSelector):
    """ A feature selection wrapper method, based on the PTA extended sequential selection

    """

    def __init__(self, threshold=0.01, l=5, r=2, score_func=GaussianNB(), cv_groups=5):
        """ Initialisation of PTA, also choosing the pick 'l' and take away 'r' variables

        :param threshold: The threshold for change in quality with new features
        :param l: The variable that shows how much features to add at most in one run.
        :param r: The variable that shows how much features to remove at most in one run
        :param score_func: The score function that computes the quality of the change
        :param cv_groups: The number of cross validation groups to be used with the score function
        """

        super().__init__(threshold=threshold, score_func=score_func, cv_groups=cv_groups)

        self.l = l
        self.r = r

    def fit(self, X, y):
        """ Fit the data of X and y to find the features to be selected.

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        # Loop through all features to find the best
        self.selected_features = np.full(X.shape[1], False, dtype=bool)

        curr_loc = 0

        # While not all locations are yet investigated, use forward and backward selection
        while curr_loc < X.shape[0]:
            curr_loc = self._forward_selection(X, y, curr_loc)
            self._backward_selection(X, y, curr_loc)

        return self

    def _forward_selection(self, X, y, curr_loc):
        """ Use forward selection to find the features to be selected

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :param curr_loc: The location to continue on in the feature set
        :return: self
        """

        curr_l = 0

        for i in range(curr_loc, X.shape[1]):

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = True

            # Compute the new score with the added feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Add feature if chance is big enough
            if new_score - self.threshold > self.score:
                self.selected_features[i] = True
                self.score = new_score
                curr_l += 1

            # Start backward selection if l are picked
            if curr_l == self.l:
                return i

        return X.shape[0] + 1

    def _backward_selection(self, X, y, curr_loc):
        """ Use backward selection to find the features to be selected

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :param curr_loc: The location to continue on in the feature set
        :return: self
        """

        curr_r = 0

        # Loop through processed features to remove the worst
        for i in reversed(range(curr_loc)):

            # If already false (useful for further implementations)
            if not self.selected_features[i]:
                continue

            # Create a new candidate feature set
            candidate_feature_set = copy.copy(self.selected_features)
            candidate_feature_set[i] = False

            # Compute the new score with the removed feature
            new_score = self._compute_score(X[:, candidate_feature_set], y)

            # Remove feature if chance is big enough
            if new_score + self.threshold > self.score:
                self.selected_features[i] = False
                self.score = new_score
                curr_r += 1

            # Stop backward selection when r are removed
            if curr_r == self.r:
                return

        return


class FloatingSelector(ForwardSelector, BackwardSelector):
    """

    """

    def __init__(self, threshold=0.01, max_iter=100, score_func=GaussianNB(), cv_groups=5):
        """ Initialisation of the wrapper selector, defining several important variables beforehand

        :param threshold: The threshold for change in quality with new features
        :param max_iter: The number of iterations the floating selector should continue
        :param score_func: The score function that computes the quality of the change
        :param cv_groups: The number of cross validation groups to be used with the score function
        """

        super(ForwardSelector, self).__init__(threshold=threshold, score_func=score_func, cv_groups=cv_groups)

        self.max_iter = max_iter

    def fit(self, X, y):
        """ Fit the data of X and y to find the features to be selected.

        :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
        :param y: A numpy array of the output. The length of the array should correspond to the size of the first
        axis of X
        :return: self
        """

        # Loop through all features to find the best
        self.selected_features = np.full(X.shape[1], False, dtype=bool)

        # Continue until the number of maximum iterations is reached
        for i in range(self.max_iter):

            current_set = copy.copy(self.selected_features)

            self._forward_selection(X, y)
            self._backward_selection(X, y)

            # If no change was present this iteration
            if np.array_equal(current_set, self.selected_features):
                return self

        return self
