# -*- coding: utf-8 -*-
"""Generic extension SelectFromModel mixin"""

# Author: T.P.A. Beishuizen <tim.beishuizen@gmail.com>


from sklearn.feature_selection import SelectFromModel

import numpy as np


class SelectKFromModel(SelectFromModel):
    """ Extends select from model by choosing k best

    """

    # variable threshold now holds k

    # Overrides by taking k best instead of threshold
    def _get_support_mask(self):
        # SelectFromModel can directly call on transform.
        if self.prefit:
            estimator = self.estimator
        elif hasattr(self, 'estimator_'):
            estimator = self.estimator_
        else:
            raise ValueError(
                'Either fit the model before transform or set "prefit=True"'
                ' while passing the fitted estimator to the constructor.')
        scores = _get_feature_importances(estimator)
        threshold = float(self.threshold)

        # Compute sorting of importances
        sorted_scores = np.argsort(scores)

        return sorted_scores >= sorted_scores.shape[0] - threshold


def _get_feature_importances(estimator):
    """Retrieve or aggregate feature importances from estimator"""
    importances = getattr(estimator, "feature_importances_", None)

    if importances is None and hasattr(estimator, "coef_"):
        if estimator.coef_.ndim == 1:
            importances = np.abs(estimator.coef_)

        else:
            importances = np.sum(np.abs(estimator.coef_), axis=0)

    elif importances is None:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)

    return importances