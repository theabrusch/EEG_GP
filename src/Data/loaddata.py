import numpy as np
import pandas as pd
from scipy.io import loadmat
import glob
import mne

def loademotiondata(subj_path):
    '''
    subj_path: Path to folder containing subject
    '''
    filepaths = glob.glob(subj_path+'*.set')
    data = dict()

    i=0
    for f in filepaths:
        # use loadmat to get some features explicitly
        print('Loading file', i+1, 'of', len(filepaths))
        matdata = loadmat(f)['EEG'][0][0]
        if i == 0:
            data['icaweights'] = matdata['icaweights']
            data['icasphere'] = matdata['icasphere']
            data['icawinv'] = (matdata['icawinv']-\
                              np.mean(matdata['icawinv'],axis =0)[np.newaxis,:])/\
                              np.std(matdata['icawinv'],axis =0)[np.newaxis,:]
            data['chanlocs'] = pd.DataFrame.from_records(matdata['chanlocs'][0])
            data['blink'] = matdata['blink']
            data['gdcomps'] = matdata['gdcomps']
            data['lateyes'] = matdata['lateyes']
            data['muscle'] = matdata['muscle']
            data['srate'] = matdata['srate']

        # use mne to filter and resample
        try:
            mne_data = mne.io.read_raw_eeglab(f, preload = True, verbose = False)
            mne_data.resample(sfreq = 200, verbose = False)
            eeg_unfiltered = mne_data[:][0]
            mne_data.filter(l_freq = 3, h_freq = 90, verbose = False)
            eeg = mne_data[:][0]

            if not 'data' in data.keys():
                data['data'] = eeg
            else:
                data['data'] = np.append( data['data'], eeg, axis = 1)
            
            if not 'unfilt' in data.keys():
                data['unfilt'] = eeg_unfiltered
            else:
                data['unfilt'] = np.append( data['unfilt'], eeg_unfiltered, axis = 1)
        except:
            print('Could not read eeg data from file', f)

        i+=1

    print('Computing ICs.')
    ics = np.matmul(data['icaweights'], data['data'])
    data['ics'] = (ics-np.mean(ics, axis =1)[:,np.newaxis])/np.std(ics, axis =1)[:,np.newaxis]
    ics_unfilt = np.matmul(data['icaweights'], data['unfilt'])
    data['ics_unfilt'] = (ics_unfilt-np.mean(ics_unfilt, axis =1)[:,np.newaxis])/np.std(ics_unfilt, axis =1)[:,np.newaxis]

    return data
