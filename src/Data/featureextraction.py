from datetime import time
from os import stat_result
import numpy as np
import pandas as pd
import mne
from scipy.stats import entropy, gaussian_kde


def temporalfeatures(data):
    '''
    Data is a dictionary containing the data for one subject as
    defined by the functions in loaddata.py. 
    '''
    features = pd.DataFrame()
    ics = data['ics']
    nsegs = int(ics.shape[1]/200)
    logRange = np.zeros((nsegs, ics.shape[0]))
    var1sAvg = np.zeros((nsegs, ics.shape[0]))
    timeEntropy = np.zeros((nsegs, ics.shape[0]))

    for i in range(nsegs):
        seg = ics[:,i*200:(i+1)*200]
        logRange[i] = np.log(np.max(seg, axis = 1) - np.min(seg, axis = 1))
        var1sAvg[i] = np.var(seg)
        for ic in range(seg.shape[0]):
            kde = gaussian_kde(seg[i,:])
            dist = kde(seg[i,:])
            timeEntropy[i, ic] = entropy(dist)

    features['logRangeTemporalVar'] = np.var(logRange, axis = 0)
    features['var1sAvg'] = np.mean(var1sAvg, axis = 0)
    features['timeEntropyAvg'] = np.mean(timeEntropy, axis = 0)

    return features


def spatialfeatures(data):
    