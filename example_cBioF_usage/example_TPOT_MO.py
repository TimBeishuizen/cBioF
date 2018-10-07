from cBioF.read_csv_dataset import read_csv_dataset as RCD

from TPOT_extension import TPOTClassifier

X, y, features = RCD('MicroOrganismsDataset.csv')

tpot = TPOTClassifier(population_size=20, generations=5, feature_selection=True, fs_modifier=0.99)

tpot.fit(X, y)

# Export pipeline
print('Exporting as Micro-Organisms pipeline.py')
tpot.export('TPOT_pipeline_MO.py')