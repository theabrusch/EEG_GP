import numpy as np 
import mne 
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from src.gp_algorithms import Multiclass_GP
from src.kernels import SquaredExponentialKernel
from scipy.linalg import fractional_matrix_power, cholesky, cho_solve, block_diag

def Algorithm33(K, y, num_classes, maxiter=100, tol=1e-6):
    n = int(len(y)/num_classes)
    f = np.zeros(len(y))
    f_temp = np.array([f[c*n:(c+1)*n] for c in range(num_classes)]).T
    i = 0
    log_marginal_likelihood = -np.inf

    while True:
        i+=1
        pi_temp = np.exp(f_temp)/np.sum(np.exp(f_temp), axis = 1)[:,np.newaxis] # eq 3.34
        pi = np.reshape(pi_temp,(num_classes*n),order= 'F')

        E = np.zeros((num_classes*n,n))
        z = np.zeros(num_classes)
        M_temp = np.zeros((n,n))

        for c in range(num_classes): 
            pi_c = pi[c*n:(c+1)*n]
            K_c = K[c*n:(c+1)*n, :]
            pi_K = np.sqrt(pi_c)[:, np.newaxis] * K_c * np.sqrt(pi_c)
            L = cholesky(np.identity(n)+pi_K, lower =True)
            temp = cho_solve((L, True), np.diag(np.sqrt(pi_c)))
            E[c*n:(c+1)*n,:] = np.sqrt(pi_c)[:, np.newaxis]*cho_solve((L, True), temp)
            M_temp = M_temp + E[c*n:(c+1)*n,:]
            z[c] = np.sum(np.log(np.diag(L)))

        M = cholesky(M_temp, lower = True)
        
        # eq. 3.39
        b = (pi-pi*pi)*f + y - pi 
        # eq. 3.39 using 3.45 and 3.47
        temp = np.reshape([np.dot(K[c*n:(c+1)*n,:],b[c*n:(c+1)*n]) \
                           for c in range(num_classes)], (num_classes*n))
        c = np.reshape([np.dot(E[c*n:(c+1)*n,:],temp[c*n:(c+1)*n]) \
                        for c in range(num_classes)], (num_classes*n)) 
        R_c = np.sum([c[i*n:(i+1)*n] for i in range(num_classes)], axis = 0).T
        temp = cho_solve((M.T, True), cho_solve((M,True), R_c))
        a = b - c + np.dot(E, temp)

        f_new = np.reshape([np.dot(K[c*n:(c+1)*n, :], a[c*n:(c+1)*n]) \
                        for c in range(num_classes)], num_classes*n) 
        diff = abs(f_new-f)
        f = f_new
        # Approximate marginal log likelihood
        f_temp = np.array([f[c*n:(c+1)*n] for c in range(num_classes)]).T
        lml = -1/2*np.sum(a*f) + np.sum(y*f) + \
              np.sum(np.log(np.sum(np.exp(f_temp), axis = 1))) - \
              np.sum(z) # eq 3.44
        
        if i==maxiter or np.mean(diff) < tol:
            log_marginal_likelihood = lml
            break
        log_marginal_likelihood = lml
    
    return f, diff, i, log_marginal_likelihood

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

SE = SquaredExponentialKernel(l = 4, sigmaf = 1)
K_temp = SE(x)
K = np.concatenate((np.concatenate((K_temp, K_temp)), K_temp)) # use the same kernel for all three classes

f, diff, i, log_marginal_likelihood = Algorithm33(K, y, 3, maxiter=50, tol=1e-10)

MC_GP = Multiclass_GP(SE, y, 3, K=K)
f, stats = MC_GP.inference()