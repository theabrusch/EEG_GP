import numpy as np
import pandas as pd
from scipy.io import loadmat
import glob
import mne

def loademotiondata(subj_path):
    '''
    subj_path: Path to folder containing subject
    '''
    filepaths = glob.glob(subj_path+'/*.set')
    data = dict()

    i=0
    for f in filepaths:
        # use loadmat to get some features explicitly
        print('Loading file', i, 'of', len(filepaths))
        matdata = loadmat(f)['EEG'][0][0]
        if i == 0:
            data['icaweights'] = matdata['icaweights']
            data['icasphere'] = matdata['icasphere']
            data['icawinv'] = matdata['icawinv']
            data['chanlocs'] = pd.DataFrame.from_records(matdata['chanlocs'][0])
            data['blink'] = matdata['blink']
            data['gdcomps'] = matdata['gdcomps']
            data['lateyes'] = matdata['lateyes']
            data['muscle'] = matdata['muscle']
            data['srate'] = matdata['srate']

        # use mne to filter and resample
        mne_data = mne.io.read_raw_eeglab(f, preload = True, verbose = False)
        mne_data.filter(l_freq = 3, h_freq = 90, verbose = False)
        mne_data.resample(sfreq = 200, verbose = False)
        eeg = mne_data[:][0]

        if not 'data' in data.keys():
            data['data'] = eeg
        else:
            data['data'] = np.append( data['data'], eeg, axis = 1)
        i+=1

    print('Computing ICs.')
    data['ics'] = np.matmul(data['icaweights'], data['data'])

    return data