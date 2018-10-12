import cBioF.TPOT_extension as TPOT
from cBioF.explore_dataset import explore_dataset
from cBioF.robustness_methods import robustness_methods


def analyse_dataset(X, y, features, file_name='Pipeline', preprocessing=False, feature_selection=False, classification=True):
    """ Analyses the dataset and exports  a file with the combination of analysis and preprocessing techniques that give
    the best results. Preprocessing can be done before analysis to prepare the dataset if needed. If not preprocessing
    is done, the data is assumed to be numeric.

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param y: A numpy array of the output. The length of the array should correspond to the size of the first
    axis of X
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the
    second axis of X
    :param file_name: The name of the file that the new pipeline will be given
    :param preprocessing: Whether preprocessing should be done before analysis
    :param feature_selection: Whether feature selection should be done. This is overridden when using preprocessing
    :param classification: Whether classification data are used. If regression data are used, preprocessing is not available
    :return:
    """

    # Test the input to be according to the standards
    robustness_methods.check_input_arrays(X, y, features)

    scoring = 'accuracy'

    # First preprocessing
    if preprocessing and classification:
        print("Preprocessing the data...")
        X, y, f, exploration_results = explore_dataset(X, y, features, preprocessing=True, classification=classification)

        feature_selection = exploration_results['fs']

        if exploration_results['imbalance']:
            scoring = 'f1'

    high_nr_features = None

    # Number of features makes a difference for the type of feature selection TPOT
    if features.shape[0] > 20000:
        high_nr_features = "TPOT FS"

    # Then analysis with automated machine learning
    if classification:
        if feature_selection:
            tpot = TPOT.TPOTClassifier(population_size=20, generations=5, scoring=scoring,
                                       feature_selection=feature_selection, fs_modifier=True,
                                       config_dict=high_nr_features)
        else:
            tpot = TPOT.TPOTClassifier(population_size=20, generations=5, scoring=scoring)
    else:
        tpot = TPOT.TPOTRegressor(population_size=20, generations=5, scoring=scoring)

    print("Using TPOT to find the best pipeline...")
    tpot.fit(X, y)

    # Export pipeline
    print('Exporting as %s.py' % file_name)
    tpot.export('%s.py' % file_name)

