from sklearn import preprocessing as PP

import numpy as np

import warnings
warnings.simplefilter('ignore')


def hot_encode_categorical_values(X, hot_encoders=None, missing_locations=None, missing_values=None):
    """ Hot encode categorical values for knn and regression imputation
    
    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param hot_encoders: Predefined hot encoders used for all values
    :param missing_locations: The locations for which the value is missing, becoming output for knn and regression
    :param missing_values: The value for missing values.
    :return: New X with categorical values hot encoded
    """



    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))

    for i in range(X.shape[1]):

        # If a column has missing values, remove the column
        if np.any(missing_locations == i) or np.any(X[:, i] == missing_values):
            continue

        # Copy a new row
        new_col = np.copy(X[:, i:i+1])

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)

        except:
            if hot_encoders is None:
                # Create hot encoder en use it for fitting and transformation
                hot_encoder = PP.MultiLabelBinarizer()
                new_col = hot_encoder.fit_transform(new_col)
            else:
                new_col = hot_encoders[i].transform(new_col)

        # Keep record of the new data and
        new_X = np.append(new_X, new_col, axis=1)

    return new_X


def scale_numerical_values(X, scalers=None, missing_locations=None, missing_values=None):
    """ Scale numeric values for knn and regression imputation

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param scalers: Predefined scalers used for all values
    :param missing_locations: The locations for which the value is missing, becoming output for knn and regression
    :param missing_values: The value for missing values.
    :return: New X with numeric values scaled
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))

    for i in range(X.shape[1]):

        # Copy a new row and delete the missing values
        new_col = np.copy(X[:, i:i + 1])
        # new_col = np.delete(new_col, new_col == missing_values, axis=0)

        if np.any(missing_locations == i) or np.any(X[:, i] == missing_values):
            new_X = np.append(new_X, new_col, axis=1)
            continue

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)

            if scalers is None:
                # Create hot encoder en use it for fitting and transformation
                scaler = PP.StandardScaler()
                new_col = scaler.fit_transform(new_col)
            else:
                new_col = scalers[i].transform(new_col)

        except:

            new_col = new_col

        # Keep record of the new data and
        new_X = np.append(new_X, new_col, axis=1)

    return new_X


def find_hot_encoders(X, missing_values=None):
    """ Find hot encoders for every feature

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param missing_values: The value for missing values.
    :return: Hot encoders to be used for future hot encoding
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))
    hot_encoders = []

    for i in range(X.shape[1]):

        # Copy a new row and delete the missing values
        new_col = np.copy(X[:, i:i + 1])
        new_col = np.delete(new_col, new_col == missing_values, axis=0)

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)
            hot_encoder = None
        except:

            # Create hot encoder en use it for fitting and transformation
            hot_encoder = PP.MultiLabelBinarizer()
            new_col = hot_encoder.fit_transform(new_col)

        # Keep record of the new data and
        hot_encoders.append(hot_encoder)

    return hot_encoders


def find_scalers(X, missing_values=None):
    """ Find scaling functions for every feature

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param missing_values: The value for missing values.
    :return: Scaling functions for future scaling
    """

    X = np.asarray(X)

    new_X = np.zeros(([X.shape[0], 0]))
    scalers = []

    for i in range(X.shape[1]):

        # Copy a new row and delete the missing values
        new_col = np.copy(X[:, i:i + 1])
        new_col = np.delete(new_col, new_col == missing_values, axis=0)

        # Find out if data is categorical
        try:
            new_col = new_col.astype(float)
            scaler = PP.StandardScaler()
            scaler.fit(new_col)
        except:

            # Create hot encoder en use it for fitting and transformation
            scaler = None

        # Keep record of the new data and
        scalers.append(scaler)

    return scalers