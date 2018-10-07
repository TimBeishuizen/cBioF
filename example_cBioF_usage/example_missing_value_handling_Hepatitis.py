from cBioF.missing_value_handling import list_deletion

from cBioF.missing_value_handling import impute
from cBioF.read_csv_dataset import read_csv_dataset as RCD

X, y, features = RCD('HepatitisDataset.csv')


# List deletion
X_cca, y_cca = list_deletion.cca(X, y, missing_values='')

# Mean Imputation
X_mean = impute.mean_imputation(X, missing_values='')

# kNN Imputation
X_knn = impute.kNN_imputation(X, missing_values='', k=3)

print("Shape change")
print("The dataset had %i samples and %i features" % (X.shape[0], X.shape[1]))
print("After cca, the dataset has %i samples" % (X_cca.shape[0]))

