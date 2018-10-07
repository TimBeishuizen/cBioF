import cBioF.TPOT_extension as TPOT
from cBioF.preprocessing import hot_encoding as HE
from cBioF import read_csv_dataset as RCD

X, y, features = RCD('HepatitisDataset.csv')

X, features = HE.hot_encode_categorical_features(X, features)

print('Exploring dataset')

tpot = TPOT.TPOTClassifier(population_size=20, generations=5)

tpot.fit(X, y)

# Export pipeline
print('Exporting as Hepatitis pipeline.py')
tpot.export('TPOT_pipeline_Hepatitis.py')
