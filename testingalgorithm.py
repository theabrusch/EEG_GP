import numpy as np 
import mne 
import matplotlib.pyplot as plt
from src.gp_algorithms import Multiclass_GP
from src.kernels import SquaredExponentialKernel, MultiClassKernel, AGDTW

## Simulate multiclass problem
mu = [[-10,-10], [0, 0], [10, 10]]
sig = 1.5
N = 10

x = np.zeros((3*N,2))
y = np.zeros((3*N,3))
for i in range(3*N):
    if i < N:
        mean = mu[0]
        y[i,0] = 1
    elif i < 2*N:
        mean = mu[1]
        y[i,1] = 1
    else:
        mean = mu[2]
        y[i,2] = 1
    x[i,:] = np.random.multivariate_normal(mean, sig*np.eye(2))

y = np.reshape(y, 3*3*N, order = 'F')

SE = MultiClassKernel(num_classes = 3, params=[[1,1.5]],\
                      base_kernel = SquaredExponentialKernel)
K = SE(x)

MC_GP = Multiclass_GP(SE, y, 3, K=K)
f, stats = MC_GP.inference()

xtest = x[[0,10,20],:]

out = MC_GP.predict(xtest, x = x)