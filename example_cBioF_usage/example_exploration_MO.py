from cBioF.read_csv_dataset import read_csv_dataset as RCD
from cBioF.explore_dataset import find_dataset_issues as FDI

X, y, features = RCD('MicroOrganismsDataset.csv')

print('Exploring dataset')

X_new, y_new, f_new, exploration_results = FDI(X, y, features, output_categorical=True, preprocessing=True,
                                               focus=False, plots=False)
