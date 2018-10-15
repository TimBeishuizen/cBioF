import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'example_cBioF_usage')) # Quick way to add parent directory

import cBioF

# Read the csv file. File should be a matrix with: 1st row = features, last column = output values. Input should be the path and name of a csv file
X, y, features = cBioF.read_csv_dataset('MicroOrganismsDataset.csv')

# Analyse the dataset with input matrix *X*, output array *y* and feature array *features*.
# *file_name* is the name of the file that will contain the best possible pipeline, as well as imports for that pipeline
# *preprocessing* should be True if preprocessing must be done beforehand
# *feature_selection* should be True if you want TPOT to focus on feature selection. This will be overridden by preprocessing
# *classification* should be True if you have classification value
pipeline = cBioF.analyse_dataset(X, y, features, file_name=None, preprocessing=False, feature_selection=True,
                                 classification=True)

print("The optimized pipeline is:")
print(pipeline)
