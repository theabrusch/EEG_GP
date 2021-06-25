import pickle
import glob
from scipy.io import loadmat
import numpy as np
import pandas as pd

pickle_paths = glob.glob("features/emotion/*.pkl")
df_collect = pd.DataFrame()
labs = ['blink', 'lateyes', 'muscle', 'heart', 'gdcomps']
for path in pickle_paths:
    df_temp = pickle.load(open(path, 'rb'))
    subj = path.split('/')[-1].split('.')[0]
    subjpath = 'SCCN_data/emotion/' + subj + '/sources.set'
    file = loadmat(subjpath)['EEG'][0][0]
    labels = np.zeros(len(df_temp))
    i = 1
    for lab in labs:
        try:
            labels[file[lab]-1] = i
        except:
            print('No', lab, 'comp for', subj)
        i+=1
    df_temp['label'] = labels.astype(np.int32)
    df_collect = df_collect.append(df_temp)

pickle.dump(df_collect, open('data_collect.pkl', 'wb'))