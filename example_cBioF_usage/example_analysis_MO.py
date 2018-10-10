from cBioF import read_csv_dataset
from cBioF import analyse_dataset

X, y, features = read_csv_dataset('MicroOrganismsDataset.csv')

# Dataset is analysed without preprocessing
analyse_dataset(X, y, features, file_name='TPOT_pipeline_MO', preprocessing=False, feature_selection=True,
                classification=True)