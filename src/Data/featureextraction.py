from datetime import time
from os import stat_result
import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from scipy.stats import entropy, gaussian_kde
from scipy.signal import spectrogram
import math

class FeatureExtractor():
    def __init__(self, data, subject):
        self.data = data
        self.subj = subject
    
    def __call__(self):
        print('Calculating features for subject', self.subj)
        print('Calculating temporal features...')
        tempfeat = self.temporalfeatures()
        print('Calculating spatial features...')
        spatfeat = self.spatialfeatures()
        print('Calculating spectral features...')
        spectfeat = self.spectralfeatures()
        print('Done calculating features for subject', self.subj)
        features = tempfeat.join(spatfeat)
        features = features.join(spectfeat)
        features['subject'] = self.subj
        return features

    def temporalfeatures(self):
        '''
        Data is a dictionary containing the data for one subject as
        defined by the functions in loaddata.py. 
        '''
        features = pd.DataFrame()
        ics = self.data['ics']
        nsegs = int(ics.shape[1]/200)
        logRange = np.zeros((nsegs, ics.shape[0]))
        var1sAvg = np.zeros((nsegs, ics.shape[0]))
        timeEntropy = np.zeros((nsegs, ics.shape[0]))

        for i in range(nsegs):
            seg = ics[:,i*200:(i+1)*200]
            logRange[i] = np.log(np.max(seg, axis = 1) - np.min(seg, axis = 1))
            var1sAvg[i] = np.var(seg)
            for ic in range(seg.shape[0]):
                kde = gaussian_kde(seg[ic,:])
                dist = kde(seg[ic,:])
                timeEntropy[i, ic] = entropy(dist)

        features['logRangeTemporalVar'] = np.var(logRange, axis = 0)
        features['var1sAvg'] = np.mean(var1sAvg, axis = 0)
        features['timeEntropyAvg'] = np.mean(timeEntropy, axis = 0)

        return features


    def spatialfeatures(self):
        '''
        Data is a dictionary containing the data for one subject as
        defined by the functions in loaddata.py. 
        '''
        spatialfeatures = pd.DataFrame()
        spatialfeatures['SAD'] = self.SAD()
        spatialfeatures['SED'] = self.SED()
        spatialfeatures['logRangeSpatial'] = self.logRangeSpatial()
        spatialfeatures['spatDistExtrema'] = self.spatDistExtrema()
        spatialfeatures['central'] = self.central()
        spatialfeatures['centralActivation'] = self.centralActivation()
        spatialfeatures['abs_med_topog'] = self.abs_med_topog()
        scalpact, virt = self.scalpact()
        spatialfeatures[virt] = scalpact

        return spatialfeatures
    
    def abs_med_topog(self):
        '''
        (absMedTopog) The absolute value of the median of the values in the scalp map.
        '''
        topog = self.data['icawinv']
        med = np.median(topog, axis = 0)
        return abs(med)
    
    def GD(self):
        '''
        (GD) Generic discontinuity measure.
        '''
    
    def scalpEntropy(self):
        '''
        (scalpEntropy) The entropy of the scalp map.
        '''

    def SAD(self):
        '''
        Calculate SAD based on electrode placement.
        (SAD) Spatial average difference (Mognon et al., 2010). 
        This feature is defined as the absolute value of the mean of 
        frontal electrode activations minus the absolute value of 
        the mean of posterior electrode activations. The frontal area 
        is defined to be the electrodes with absolute angles less 
        than 60° and radii larger than 0.4. The posterior area 
        consists of the electrodes with absolute angles larger than 
        110°.
        '''
        frontalelec = ((abs(self.data['chanlocs']['theta']) < 60) & \
                    (self.data['chanlocs']['radius'] > 0.40))
        backelec = abs(self.data['chanlocs']['theta']) > 110
        frontact = abs(np.mean(self.data['icawinv'][frontalelec,:], axis = 0))
        backact = abs(np.mean(self.data['icawinv'][backelec,:], axis = 0))

        SAD = frontact - backact

        return SAD

    def SED(self):
        '''
        (SED) Spatial eye difference (Mognon et al., 2010). Absolute 
        value of the difference between activation of electrodes around 
        the left and right eye areas. The left eye area is defined to 
        lie between the angles −61° and −35° with a radius larger than 
        0.3 (where the head radius is assumed to be one, the convention 
        in EEGLAB). The right eye area is defined to lie between the 
        angles 34° and 61°, also at a radius larger than 0.3. Zero 
        degrees is towards the nose and positive 90° is at the right 
        ear.
        '''

        lefteye = ((self.data['chanlocs']['theta'] > -61) & \
                (self.data['chanlocs']['theta'] < -35) & \
                (self.data['chanlocs']['radius'] > 0.30))
        righteye = ((self.data['chanlocs']['theta'] > 34) & \
                (self.data['chanlocs']['theta'] < 61) & \
                (self.data['chanlocs']['radius'] > 0.30))
        
        lefteyeact = np.mean(self.data['icawinv'][lefteye,:], axis = 0)
        righteyeact = np.mean(self.data['icawinv'][righteye,:], axis = 0)
        SED = abs(lefteyeact-righteyeact)

        return SED

    def logRangeSpatial(self):
        '''
        (logRangeSpatial) Logarithm of range of activation of 
        electrodes. This was one of the six final features included in 
        the classifier described in Winkler et al., 2011.
        '''
        minact = np.min(self.data['icawinv'], axis = 0)
        maxact = np.max(self.data['icawinv'], axis = 0)

        return np.log(maxact-minact)

    def spatDistExtrema(self):
        '''
        (spatDistExtrema) Euclidean distance in 3D coordinates between
        the two electrodes with minimal and maximal activation.
        '''

        minactarg = np.argmin(self.data['icawinv'], axis = 0)
        maxactarg = np.argmax(self.data['icawinv'], axis = 0)

        minlocs = self.data['chanlocs'][['X', 'Y', 'Z']].take(minactarg).values
        maxlocs = self.data['chanlocs'][['X', 'Y', 'Z']].take(maxactarg).values

        #Compute 2-norm
        dist = maxlocs-minlocs 
        temp = (dist*dist).astype(np.float32)
        spatDistExtrema = np.sqrt(temp[:,0]+temp[:,1]+temp[:,2])
        return np.log(spatDistExtrema)



    def scalpact(self):
        '''
        (central (Cz), frontal (AFz), posterior (POz), left (C5), right (C4)) 
        These features give the absolute values of the mean 
        activations of electrodes in various areas of the scalp. Each 
        area is defined as the mean over all electrodes, where the 
        contribution from each electrode to the mean is weighted by a 
        Gaussian bell. For the areas around the eyes (lefteye and 
        righteye), the standard deviation of the Gaussian bell is 1 
        cm. For all other areas, it is 2 cm. A 9-cm radius of the scalp 
        is assumed. The Gaussian bell is centered at Cz.
        '''
        placements = ['central', 'frontal', 'posterior', 'left', 'right']
        sph_thetas = [0, 68.39, 68.39, -71.982, 48]
        sph_phis = [90, 90, -90, 0, 0]
        sigma = 2
        activations = np.zeros((len(self.data['icawinv']), len(sph_thetas)))
        
        for i in range(len(sph_thetas)):
            phi = sph_phis[i]
            theta = sph_thetas[i]
            activations[:,i] = self.virtualactivation(phi, theta, sigma,\
                                                      self.data['chanlocs'],\
                                                      self.data['icawinv'])
        
        return abs(activations), placements
        
    def spherical_distance(self, sph_phi1, sph_theta1, sph_phi2, sph_theta2):
        lambdaf = sph_phi1/180*np.pi # longitude
        lambdas = sph_phi2/180*np.pi # longitude
        phif = sph_theta1/180*np.pi #latitude
        phis = sph_theta2/180*np.pi #latitude
        delta_lambda = lambdaf-lambdas
        cosphif = np.cos(phif)
        cosphis = np.cos(phis)
        sinphif = np.sin(phif)
        sinphis = np.sin(phis)
        sph_radius = 9
        
        central_angle = np.zeros(len(phis))
        # http://en.wikipedia.org/wiki/Great-circle_distance
        for i in range(len(phis)): 
            central_angle[i] = math.atan2(np.sqrt((cosphif*np.sin(delta_lambda[i]))**2\
                                        +(cosphis[i]*sinphif - \
                                    sinphis[i]*cosphif*np.cos(delta_lambda[i]))**2),\
                                    (sinphis[i]*sinphif+cosphis[i]*cosphif*np.cos(delta_lambda[i])))

        distance = (sph_radius*central_angle)

        return distance

    def virtualactivation(self, phis, thetas, sigmas, chanlocs, scalpmap):
        all_electrodes_phi = chanlocs['sph_phi'].explode().explode().values.astype(np.float32)
        all_electrodes_theta = chanlocs['sph_theta'].explode().explode().values.astype(np.float32)
        sph_radius = chanlocs['sph_radius'].explode().explode().values.astype(np.float32)

        distances = self.spherical_distance(phis, thetas, all_electrodes_phi, \
                                            all_electrodes_theta, sph_radius)
        distances = distances**2
        exp_distances = np.exp(-distances/(2*sigmas**2))
        exp_distances=exp_distances/np.sum(exp_distances)
        activations = np.sum(scalpmap*exp_distances[:,np.newaxis], axis = 0)
        
        return activations

    def centralActivation(self):
        '''
        (centralActivation) Logarithm of mean of absolute values of
        activations of central electrodes of IC (Winkler et al., 2011).
        '''
        central_theta = 0
        central_phi = np.pi/2
        sph_phi = self.data['chanlocs']['sph_phi'].explode().explode().values.astype(np.float32)
        sph_theta = self.data['chanlocs']['sph_theta'].explode().explode().values.astype(np.float32)
        lambdas = sph_phi/180*np.pi # longitude
        phif = sph_theta/180*np.pi #latitude
        dist = np.sqrt((central_theta-sph_theta)**2 + \
                    (central_phi-sph_phi)**2)
        sorted = np.argsort(dist)
        centralgroup = sorted[0:13]
        icaact = self.data['icawinv']
        centralelec = icaact[centralgroup,:]

        centralActivation = np.log(np.mean(abs(centralelec), axis = 0))
        return centralActivation

    def dipoleFeatures(self):
        '''
        (dipoleResidualVariance) Residual variance of dipole fit.
        (zcoord) X,Y,and Z coordinates of dipole fit.
        '''
    def cdn(self):
        '''
        (cdn) Current density norm (Winkler et al., 2011). The current 
        density norm is a measure of the complexity of the current 
        source distribution of an IC. A high complexity of the current 
        source distribution indicates that the source of the IC is 
        difficult to locate inside the brain, and thus that it is likely 
        to be an artifact. This was one of the six final features 
        included in the classifier described in Winkler and colleagues 
        (2011), in which a more detailed description can be found.
        '''
        m100 = loadmat('src/Data/dipolfit_matrix.mat')
        M100 = m100['M100']
        clab_temp = m100['clab'][0]
        clab = np.array([elec[0].lower() for elec in clab_temp])
        
        channel_labels=np.array([lab[0].lower() for lab in self.data['chanlocs']['labels']])

        print('hej')


    def spectralfeatures(self):
        '''
        (theta) Mean over 1-s intervals of the logarithm of band power 
        in the θ (4–7 Hz) band.

        (lowFrequentPowerAvg) This features give the band power in the 
        δ band (1–3 Hz) relative to the total power in the time series. 
        The spectrogram used for these features is calculated based on 
        the downsampled but unfiltered time series since the filter 
        removes frequencies lower than 3 Hz. The spectrogram is 
        calculated over 1-s intervals, and the power in the δ band 
        divided by the power over all frequencies is then found. The 
        feature lowFrequentPowerAvg is the mean over the 1-s intervals 
        of this ratio.
        '''
        spectralfeatures = pd.DataFrame()
        ics = self.data['ics_unfilt']
        thetapower = np.zeros(ics.shape[0])
        lowFrequentPowerAvg = np.zeros(ics.shape[0])

        thetarange = [4, 7]
        deltarange = [1, 3]

        for ic in range(ics.shape[0]):
            f, _ , spect = spectrogram(ics[ic,:], fs = 200, nperseg = 200, noverlap = 100)
            spectout = spect*spect
            thetaband = (f>=thetarange[0]) & (f<=thetarange[1])
            thetapower[ic] = np.mean(np.log(np.mean(spectout[thetaband,:], axis = 0)))

            deltaband = (f>=deltarange[0]) & (f<=deltarange[1])
            deltarelpow =  np.mean(spectout[deltaband,:], axis = 0) / np.mean(spectout, axis = 0)
            lowFrequentPowerAvg[ic] = np.mean(deltarelpow)
        
        spectralfeatures['theta'] = thetapower
        spectralfeatures['lowFrequentPowerAvg'] = lowFrequentPowerAvg

        return spectralfeatures