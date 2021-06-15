from datetime import time
import numpy as np
import pandas as pd
import mne
from scipy.stats import entropy¶


def temporalfeatures(data):
    '''
    Data is a dictionary containing the data for one subject as
    defined by the functions in loaddata.py. 
    '''
    features = pd.DataFrame()
    ics = data['ics']
    nsegs = int(ics.shape[1]/200)
    logRange = np.zeros(nsegs)
    var1sAvg = np.zeros(nsegs)
    timeEntropy = np.zeros(nsegs)

    for i in range(nsegs):
        seg = ics[:,i*200:(i+1)*200]
        logRange[i] = np.log(np.max(seg, axis = 1) - np.min(seg, axis = 1))
        var1sAvg[i] = np.var(seg)
        timeEntropy[i] = entropy(seg, axis = 1)

    features['logRangeTemporalVar'] = np.var(np.log(logRange), axis = 0)
    features['var1sAvg'] = np.mean(var1sAvg, axis = 0)
    features['timeEntropyAvg'] = np.mean(timeEntropy, axis = 0)

    return features

