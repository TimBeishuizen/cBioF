import cBioF

X, y, features = cBioF.read_csv_dataset('MicroOrganismsDataset.csv')

print('Exploring dataset')

X_new, y_new, f_new, exploration_results = cBioF.find_dataset_issues(X, y, features, output_categorical=True,
                                                                     preprocessing=True, focus=False, plots=False)
