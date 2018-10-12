import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.04583703480408602
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=45),
    RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.35000000000000003, n_estimators=100), step=0.9500000000000001),
    Binarizer(threshold=0.8),
    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.25, min_samples_leaf=19, min_samples_split=19, n_estimators=100)),
    LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
