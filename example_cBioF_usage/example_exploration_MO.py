import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'example_cBioF_usage')) # Quick way to add parent directory

import cBioF

# Read the csv file. File should be a matrix with: 1st row = features, last column = output values. Input should be the path and name of a csv file
X, y, features = cBioF.read_csv_dataset('MicroOrganismsDataset.csv')

# Explore the dataset with input matrix *X*, output array *y* and feature array *features*.
# *classification* should be True if the output values are categorical,
# *preprocessing* should be True if also preprocessing should be done,
# *focus* should be True if also outliers for several issues should be shown and
# *plots* should be True if also additional plots of distributions should be shown.
X_new, y_new, f_new, exploration_results = cBioF.explore_dataset(X, y, features, classification=True,
                                                                 preprocessing=True, focus=False, plots=False)

# Export the newly preprocessed values of the dataset
cBioF.export_csv_dataset(X_new, y_new, f_new, csv_path='MicroOrganismsDatasetNew.csv')

