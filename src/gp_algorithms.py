
from scipy.linalg import cholesky, cho_solve
import numpy as np 


class Multiclass_GP():

    def __init__(self, kernel, targets, C, K=None, x = None):
        self.kernel = kernel
        self.y = targets
        self.num_classes = C
        self.f = None
        self.x = x

        if K is None:
            try:
                self.K = self.kernel(x)
            except ValueError:
                print('x or K must be not None')
        else:
            self.K = K

    def inference(self, maxiter=100, tol=1e-10):

        # Initialization
        self.n = int(len(self.y)/self.num_classes)
        f = np.zeros(len(self.y))
        f_temp = np.array([f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
        i = 0
        log_marginal_likelihood = -np.inf
        converged = False
        while not converged and i < maxiter:
            i+=1
            pi_temp = np.exp(f_temp)/np.sum(np.exp(f_temp), axis = 1)[:,np.newaxis] # eq 3.34
            pi = np.reshape(pi_temp,(self.num_classes*self.n),order= 'F')

            E = np.zeros((self.num_classes*self.n,self.n))
            z = np.zeros(self.num_classes)
            M_temp = np.zeros((self.n,self.n))

            for c in range(self.num_classes): 
                pi_c = pi[c*self.n:(c+1)*self.n]
                K_c = self.K[c*self.n:(c+1)*self.n, :]
                pi_K = np.sqrt(pi_c)[:, np.newaxis] * K_c * np.sqrt(pi_c)
                L = cholesky(np.identity(self.n)+pi_K, lower =True)
                temp = cho_solve((L, True), np.diag(np.sqrt(pi_c)))
                E[c*self.n:(c+1)*self.n,:] = np.sqrt(pi_c)[:, np.newaxis]*temp
                M_temp = M_temp + E[c*self.n:(c+1)*self.n,:]
                z[c] = np.sum(np.log(np.diag(L)))

            M = cholesky(M_temp, lower = True)
            
            # eq. 3.39
            b = (pi-pi*pi)*f + self.y - pi 
            # eq. 3.39 using 3.45 and 3.47
            temp = np.reshape([np.dot(self.K[c*self.n:(c+1)*self.n,:],\
                                      b[c*self.n:(c+1)*self.n]) \
                               for c in range(self.num_classes)], \
                             (self.num_classes*self.n))
            c = np.reshape([np.dot(E[c*self.n:(c+1)*self.n,:],\
                                   temp[c*self.n:(c+1)*self.n]) \
                            for c in range(self.num_classes)], \
                           (self.num_classes*self.n)) 
            R_c = np.sum([c[i*self.n:(i+1)*self.n] for i in range(self.num_classes)], axis = 0).T
            a = b - c + np.dot(E, cho_solve((M,True), R_c))

            f = np.reshape([np.dot(self.K[c*self.n:(c+1)*self.n, :],\
                                       a[c*self.n:(c+1)*self.n]) \
                                for c in range(self.num_classes)], \
                               self.num_classes*self.n) 
            
            # Approximate marginal log likelihood
            f_temp = np.array([f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
            lml = -1/2*np.sum(a*f) + np.sum(self.y*f) + \
                np.sum(np.log(np.sum(np.exp(f_temp), axis = 1))) - \
                np.sum(z) # eq 3.44
            
            if abs(lml-log_marginal_likelihood)<tol:
                converged = True
                log_marginal_likelihood = lml
                self.f = f
                break
            log_marginal_likelihood = lml
        
        stats = dict()
        stats['lml'] = log_marginal_likelihood
        stats['iter'] = i
        stats['info'] = converged

        return f, stats


    def predict(self, xstar, x = None, S=100):
        if self.f is None:
            ValueError('Run inference before prediction')
        
        if self.x is None and x is None:
            ValueError('The training input must be given at', \
                       'initialization of the class or when calling predict.')

        # Initial computations
        f_temp = np.array([self.f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
        pi_temp = np.exp(f_temp)/np.sum(np.exp(f_temp), axis = 1)[:,np.newaxis] # eq 3.34
        pi = np.reshape(pi_temp,(self.num_classes*self.n),order= 'F')
        M_temp = np.zeros((self.n, self.n))
        E = np.zeros((self.num_classes*self.n,self.n))

        if self.x is not None:
            kstar = self.kernel(x, xstar)
        else:
            kstar = self.kernel(self.x, xstar)

        for c in range(self.num_classes):
            pi_c = pi[c*self.n:(c+1)*self.n]
            K_c = self.K[c*self.n:(c+1)*self.n, :]
            pi_K = np.sqrt(pi_c)[:, np.newaxis] * K_c * np.sqrt(pi_c)
            L = cholesky(np.identity(self.n)+pi_K, lower =True)
            temp = cho_solve((L, True), np.diag(np.sqrt(pi_c)))
            E[c*self.n:(c+1)*self.n,:] = np.sqrt(pi_c)[:, np.newaxis]*temp
            M_temp = M_temp + E[c*self.n:(c+1)*self.n,:]
        
        M = cholesky(M_temp, lower = True)
        mu_star = np.zeros(len(xstar)*self.num_classes)
        Sigma = np.zeros((self.num_classes, self.num_classes))
        kstarstar = self.kernel(xstar)

        for c in range(self.num_classes):
            pi_c = pi[c*self.n:(c+1)*self.n]
            y_c = self.y[c*self.n:(c+1)*self.n]
            kstar_c = kstar[c*self.n:(c+1)*self.n]
            mu_star_c = np.sum((y_c-pi_c)*kstar_c)
            mu_star[c] = mu_star_c
            E_c = E[c*self.n:(c+1)*self.n,:]
            b = np.dot(E_c, kstar_c)
            c_def = np.dot(E_c, cho_solve((M, True), b))

            for c_mark in range(self.num_classes):
                Sigma[c,c_mark] = np.sum(c_def*kstar[c_mark*self.n:(c_mark+1)*self.n])
            
            k_starstar_c = kstarstar[c]
            Sigma[c,c] = Sigma[c,c] + k_starstar_c - np.sum(b*kstar_c)

        pistar = np.zeros((self.num_classes))

        for i in range(S):
            fstar = np.random.multivariate_normal(mu_star, Sigma)
            pistar = pistar + np.exp(fstar)/np.sum(np.exp(fstar))
        
        pistar = pistar / S

        return pistar