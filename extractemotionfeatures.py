from src.Data.loaddata import loademotiondata
from src.Data.featureextraction import FeatureExtractor
import pickle
import glob
import time

start = time.time()

subjfolders = glob.glob("SCCN_data/emotion/*/")

for subj_fold in subjfolders:
    data = loademotiondata(subj_fold)
    subj = subj_fold.split('/')[-2]
    FE = FeatureExtractor(data, subject = subj)
    features = FE()
    filepath = 'features/emotion/' + subj + '.pkl'
    pickle.dump(features, open(filepath, 'wb'))
end = time.time()

print('Elapsed time', end-start, 's')

