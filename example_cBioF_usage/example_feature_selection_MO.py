# Warnings are ignored due to cross validation of 20 classes in
import warnings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'example_cBioF_usage')) # Quick way to add parent directory

from cBioF.feature_selection import order_methods as OM, embedded_methods as EM
from cBioF.preprocessing import normalisation as NS
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from cBioF.feature_selection import wrapper_methods as WM
from cBioF.read_csv_dataset import read_csv_dataset as RCD

warnings.simplefilter('ignore')

X, y, features = RCD('MicroOrganismsDataset.csv')

X = NS.normalise_numeric_features(X.astype(float), standardisation=True)

# Example Filter methods
filter = SelectKBest(score_func=f_classif, k=150)
X_filter = filter.fit_transform(X, y)
f_filter = features[filter.get_support()]

# Example Wrapper methods - first ordering than selection
orderer = OM.FeatureOrderer(score_func=f_classif)
X_ordered = orderer.fit_transform(X, y)

wrapper = WM.ForwardSelector(threshold=0.0001)
X_wrapper = wrapper.fit_transform(X_ordered, y)
f_wrapper = features[wrapper.get_support()]

# Example Embedded methods
embedded = EM.SelectKFromModel(estimator=RandomForestClassifier(), threshold=150)
X_embedded = embedded.fit_transform(X, y)
f_embedded = features[embedded.get_support()]

print("The filter method has %i feature selected" % f_filter.shape[0])
print("The wrapper method has %i feature selected" % f_wrapper.shape[0])
print("The embedded method has %i feature selected" % f_embedded.shape[0])