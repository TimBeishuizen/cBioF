import csv
import numpy as np


def export_csv_dataset(X, y, features, csv_path):
    """ Export the data, output and features into a csv-file. This file is based on the following description:
    - Rows correspond to instances and columns correspond to features
    - The first entry every column corresponds to the name of the feature with all other entries corresponding to
    values of that feature
    - The last entry of every row corresponds to the output labels of every instance.

    :param X: A numpy matrix of the data. First axis corresponding to instances, second axis corresponding to samples
    :param y: A numpy array of the output. The length of the array should correspond to the size of the first
    axis of X
    :param features: A numpy array of the feature names. The length of the array should correspond to the size of the
    second axis of X
    :param csv_path: The location and name of the csv-file
    :return: The data in a matrix, the output labels and feature names
    """

    matrix = []

    matrix.append(features.tolist() + ['Output'])

    for i in range(y.shape[0]):
        matrix.append(X[i, :].tolist() + [y[i]])

    # Opening CSV file
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(matrix)