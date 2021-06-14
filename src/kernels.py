import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class AGDTW():
    def __init__(self, sigma, **kwargs):
        self.sigma = sigma
    
    def __call__(self, x, xstar = None):
        if xstar is not None:
            K = np.zeros((len(x), len(xstar)))
            for i in range(len(xstar)):
                print('Observation', i)
                for j in range(len(x)):
                    if (j-i)%20 == 0:
                        print(j-i, 'out of', len(x)-i)
                    _, path = fastdtw(xstar[i,:], x[j,:], dist=euclidean)
                    path = np.array(path)
                    dist = np.exp(-(xstar[i,path[:,0]] - x[j,path[:,1]])**2/self.sigma**2)
                    K[j,i] = np.sum(dist)

        else:
            K = np.zeros((len(x), len(x)))
            for i in range(len(x)):
                print('Observation', i)
                for j in range(i, len(x)):
                    if (j-i)%20 == 0:
                        print(j-i, 'out of', len(x)-i)
                    _, path = fastdtw(x[i,:], x[j,:], dist=euclidean)
                    path = np.array(path)
                    dist = np.exp(-(x[i,path[:,0]] - x[j,path[:,1]])**2/self.sigma**2)
                    K[j,i] = np.sum(dist)
                    
            # add lower triangle
            K = K + K.T - np.diag(K)
        return K


class SquaredExponentialKernel():
    def __init__(self, l, sigmaf, **kwargs):
        self.l = l
        self.sigmaf = sigmaf
        
    def __call__(self, x, xstar = None):
        
        if xstar is not None:
            K = np.zeros((len(x), len(xstar)))

            for i in range(len(xstar)):
                temp = np.repeat(np.reshape(xstar[i,:],(1,-1)), len(x), axis = 0)
                dist = np.sum((x-temp)**2, axis = 1)
                K[:,i] = self.sigmaf**2*np.exp(-dist/(2*(self.l**2)))
        else:
            K = np.zeros((len(x), len(x)))

            for i in range(len(x)):
                temp = np.repeat(np.reshape(x[i,:],(1,-1)), len(x), axis = 0)
                dist = np.sum((x-temp)**2, axis = 1)
                K[:,i] = self.sigmaf**2*np.exp(-dist/(2*(self.l**2)))
            
        return K

class MultiClassKernel():
    '''
    Wrapper for turning standard kernels into multiclass kernels
    '''
    def __init__(self, num_classes, params, base_kernel):
        self.num_classes = num_classes
        self.params = params
        self.base_kernel = base_kernel

        if not (len(self.params) == 1 or len(self.params) == self.num_classes):
            raise ValueError('The number of sets of parameters must be either 1',\
                             'or equal to the number of classes.')
    
    def __call__(self, x, xstar = None):

        if len(self.params) == 1: # use the same kernel for all classes
            kernel = self.base_kernel(*self.params[0])
            K_temp = kernel(x = x, xstar = xstar)
            K = K_temp
            for c in range(self.num_classes-1):
                K = np.concatenate((K, K_temp), axis = 0)
        elif len(self.params) == self.num_classes:
            kernel = self.base_kernel(*self.params[0])
            K = kernel(x,xstar)
            for c in range(1, self.num_classes):
                kernel = self.base_kernel(*self.params[c])
                K_temp = kernel(x,xstar)
                K = np.concatenate((K, K_temp), axis = 0)

        return K


