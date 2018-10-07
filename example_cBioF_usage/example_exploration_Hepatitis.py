from cBioF.read_csv_dataset import read_csv_dataset as RCD

from builtin_functions.preprocessing import hot_encoding as HE

from TPOT_extension import TPOTClassifier

X, y, features = RCD('HepatitisDataset.csv')

X, features = HE.hot_encode_categorical_features(X, features)

print('Exploring dataset')

tpot = TPOTClassifier(population_size=20, generations=5)

tpot.fit(X, y)

# Export pipeline
print('Exporting as Hepatitis pipeline.py')
tpot.export('TPOT_pipeline_Hepatitis.py')
