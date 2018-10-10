from cBioF import read_csv_dataset
from cBioF import analyse_dataset
import numpy as np

X, y, features = read_csv_dataset('HepatitisDataset.csv')

# Dataset is analysed with preprocessing
analyse_dataset(X, y, features, file_name='TPOT_pipeline_Hepatitis', preprocessing=True, classification=True)