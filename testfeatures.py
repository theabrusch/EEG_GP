from src.Data.loaddata import loademotiondata
from src.Data.featureextraction import FeatureExtractor
import pickle
from scipy.signal import spectrogram
import glob

subjfolders = glob.glob("SCCN_data/emotion/*/")

data = pickle.load(open('tempdata.pickle', 'rb'))

FE = FeatureExtractor(data, subject = 'eb79')
cdn = FE.cdn()
for subj_fold in subjfolders:
    data = loademotiondata(subj_fold)
    subj = subj_fold.split('/')[-2]
    FE = FeatureExtractor(data, subject = subj)
    features = FE()
    filepath = 'features/emotion/' + subj 
    pickle.dump(features, open(filepath, 'wb'))
