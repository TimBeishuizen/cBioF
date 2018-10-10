import numpy as np
import pandas as pd

def check_input_arrays(X, y, features):
    """ Check if the values for X, y an features are according to standards

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param y: A numpy array of the output. The length of the array should correspond to the size of the first
    axis of X
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the
    second axis of X
    :return: Error if not according to standard
    """

    # Check if numpy arrays
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise ValueError("The input matrix X is not a two dimensional numpy array")
    elif type(y) != np.ndarray or len(y.shape) != 1:
        raise ValueError("The output array y is not a one dimensional numpy array")
    elif type(features) != np.ndarray or len(features.shape) != 1:
        raise ValueError("The feature array features is not a one dimensional numpy array")

    # Check if shapes correspond
    if X.shape[0] != y.shape[0]:
        raise ValueError("The input matrix X and output array y do not have the same number of instances")
    elif X.shape[1] != features.shape[0]:
        raise ValueError("The input matrix X and feature array do not have the same number of features")


def check_pandas_input(DfX, Dfy):
    """ Check if the values for pandas input Dfx and dfy are according to standards

    :param DfX: A pandas dataframe of the data. Every column corresponds to a feature, rows correspond to instances.
    :param Dfy: A pandas series of the output. The length of the array should correspond to the size of the first
    axis of X
    :return: Error if not according to standard
    """

    # Check if pandas dataframes
    if type(DfX) != pd.DataFrame:
        raise ValueError("The input dataframe Dfx is not a pandas dataframe")
    elif type(Dfy) != pd.Series or len(Dfy.shape) != 1:
        raise ValueError("The output dataframe Dfx is not a pandas one dimensional series")

    # check if shapes correspond:
    if DfX.shape[0] != Dfy.shape[0]:
        raise ValueError("The input dataframe and output series do not have the same number of instances")