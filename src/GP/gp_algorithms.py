
from scipy.linalg import cholesky, cho_solve
import numpy as np 
from scipy.optimize import fminbound, brent
import warnings

class Multiclass_GP():

    def __init__(self, kernel, targets, C, K=None, x = None):
        self.kernel = kernel
        self.y = targets
        self.num_classes = C
        self.f = None
        self.x = x
        self.converged = False

        if K is None:
            try:
                self.K = self.kernel(x)
            except ValueError:
                print('x or K must be not None')
        else:
            self.K = K

    def inference(self, maxiter=100, tol=1e-6, f_init = None):

        # Initialization
        self.n = int(len(self.y)/self.num_classes)
        if f_init is None:
            f = np.zeros(len(self.y))
        else:
            f = f_init
        f_temp = np.array([f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
        i = 0
        log_marginal_likelihood = -np.inf
        converged = False
        a_old = np.zeros(f.shape)

        print('Beginning inference...')
        while not converged and i < maxiter:
            i+=1
            print('Iteration',i)
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
            temp = pi*f
            for c in range(self.num_classes):
                temp_c = np.zeros(pi.shape)
                idx = self.num_classes*self.n - c*self.n
                temp_c[:idx] = pi[c*self.n:]
                temp_c[idx:] = pi[:c*self.n]
                temp_c = temp_c*pi

                f_c = np.zeros(f.shape)
                f_c[:idx] = f[c*self.n:]
                f_c[idx:] = f[:c*self.n]
                temp = temp - f_c*temp_c 

            b = temp + self.y - pi 
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
            a_new = b - c + np.dot(E, cho_solve((M,True), R_c))
            d_a = a_new - a_old

            # line search
            def psi_line(s, a_old, d_a):
                a = a_old + s*d_a
                f = np.reshape([np.dot(self.K[c*self.n:(c+1)*self.n, :],\
                                    a[c*self.n:(c+1)*self.n]) \
                                for c in range(self.num_classes)], \
                            self.num_classes*self.n) 
                # Approximate marginal log likelihood
                f_temp = np.array([f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
                obj = 1/2*np.sum(a*f) - np.sum(self.y*f) + \
                    np.sum(np.log(np.sum(np.exp(f_temp), axis = 1)))
            
                return obj
            
            s = brent(psi_line, args = (a_old,d_a), brack = (0,2), \
                      tol = 1e-4, maxiter = 10)
            a = a_old + s*d_a
            #update f based on new a
            f = np.reshape([np.dot(self.K[c*self.n:(c+1)*self.n, :],\
                                    a[c*self.n:(c+1)*self.n]) \
                                for c in range(self.num_classes)], \
                            self.num_classes*self.n) 
            
            f_temp = np.array([f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
            # Approximate marginal log likelihood
            lml = -psi_line(s, a_old, d_a)
            a_old = a
                
            if abs(lml-log_marginal_likelihood)<tol:
                converged = True
                log_marginal_likelihood = lml - np.sum(z)
                self.f = f
                self.converged = True
                break
            log_marginal_likelihood = lml
        
        if not converged:
            warnings.warn('Inference has not converged. Predictions may be uncertain.', UserWarning)
            self.f = f

        print('Ending inference with exit status converged:', converged)
        stats = dict()
        stats['lml'] = log_marginal_likelihood
        stats['iter'] = i
        stats['info'] = converged

        return f_temp, stats

    def predict(self, xstar, x = None, S=100):
        if self.f is None:
            raise ValueError('Run inference before prediction')
        if self.x is None and x is None:
            raise ValueError('The training input must be given at', \
                              'initialization of the class or when calling predict.')
        if not self.converged:
            warnings.warn('Inference did not converge. Running predictions'+\
                          'with unstable f.')
        # Initial computations
        f_temp = np.array([self.f[c*self.n:(c+1)*self.n] for c in range(self.num_classes)]).T
        pi_temp = np.exp(f_temp)/np.sum(np.exp(f_temp), axis = 1)[:,np.newaxis] # eq 3.34
        pi = np.reshape(pi_temp,(self.num_classes*self.n),order= 'F')
        M_temp = np.zeros((self.n, self.n))
        E = np.zeros((self.num_classes*self.n,self.n))
        n_test = len(xstar)

        print('Calculating kernel between test input and training set.')
        if self.x is not None:
            kstar = self.kernel(self.x, xstar)
        else:
            kstar = self.kernel(x, xstar)
        print('Calculating kernel for test input.')
        kstarstar = self.kernel(xstar)

        print('Calculating predictive distribution.')
        for c in range(self.num_classes):
            pi_c = pi[c*self.n:(c+1)*self.n]
            K_c = self.K[c*self.n:(c+1)*self.n, :]
            pi_K = np.sqrt(pi_c)[:, np.newaxis] * K_c * np.sqrt(pi_c)
            L = cholesky(np.identity(self.n)+pi_K, lower =True)
            temp = cho_solve((L, True), np.diag(np.sqrt(pi_c)))
            E[c*self.n:(c+1)*self.n,:] = np.sqrt(pi_c)[:, np.newaxis]*temp
            M_temp = M_temp + E[c*self.n:(c+1)*self.n,:]
        
        M = cholesky(M_temp, lower = True)
        mu_star = np.zeros((len(xstar),self.num_classes))
        Sigma = np.zeros((n_test, self.num_classes, self.num_classes))

        # Infer mean and covariance for all test data points
        for c in range(self.num_classes):
            pi_c = pi[c*self.n:(c+1)*self.n, np.newaxis]
            y_c = self.y[c*self.n:(c+1)*self.n, np.newaxis]
            kstar_c = kstar[c*self.n:(c+1)*self.n]
            mu_star_c = np.sum((y_c-pi_c)*kstar_c, axis = 0)
            mu_star[:,c] = mu_star_c
            E_c = E[c*self.n:(c+1)*self.n,:]
            b = np.dot(E_c, kstar_c)
            c_def = np.dot(E_c, cho_solve((M, True), b))

            for c_mark in range(self.num_classes):
                Sigma[:,c,c_mark] = np.sum(c_def*kstar[c_mark*self.n:(c_mark+1)*self.n], axis = 0)
            
            k_starstar_c = kstarstar[c,:]
            Sigma[:,c,c] = Sigma[:,c,c] + k_starstar_c - np.sum(b*kstar_c, axis = 0)

        #MC sampling
        pistar = np.zeros((n_test, self.num_classes))
        print('Beginning MC sampling.')
        for i in range(S):
            for n in range(n_test):
                fstar = np.random.multivariate_normal(mu_star[n,:], Sigma[n,:,:])
                pistar[n,:] = pistar[n,:] + np.exp(fstar)/np.sum(np.exp(fstar))
        
        pistar = pistar / S
        dist = dict()
        dist['mean'] = mu_star
        dist['Cov'] = Sigma

        return pistar, dist