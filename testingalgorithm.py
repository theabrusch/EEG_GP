import numpy as np 
import mne 
import matplotlib.pyplot as plt
from src.GP.gp_algorithms import Multiclass_GP
from src.GP.kernels import SquaredExponentialKernel, MultiClassKernel, AGDTW
import pandas as pd

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
f, stats = MC_GP.inference(maxiter = 10)

xtest = x[[0,10,20],:]

out = MC_GP.predict(xtest, x = x)

## Test on multiclass hand written digits set
test = open('GP/zip.test')
train = open('GP/zip.train')
x_test = []
y_test = []
x_train = []
y_train = []

for line in test:
    numbers = line.split(' ')
    obs = np.zeros(len(numbers)-1)
    j = 0
    for num in numbers[1::]:
        temp = num.split("\n")
        obs[j] = float(temp[0])
        j+=1
    x_test.append(obs)
    y_test.append(float(numbers[0]))

for line in train:
    numbers = line.split(' ')
    obs = np.zeros(len(numbers)-2)
    j = 0
    for num in numbers[1::]:
        if not num == '\n':
            obs[j] = float(num)
        j+=1
    x_train.append(obs)
    y_train.append(float(numbers[0]))

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.array(x_train)
y_train = np.array(y_train)

collect = np.append(x_train, x_test, axis = 0)
y_collect = np.append(y_train, y_test, axis = 0)
idx_x_train = np.random.choice(list(range(len(collect))), size = int(len(collect)/2))
x_train_new = collect[idx_x_train,:]
x_test_new = collect[~idx_x_train,:]
y_train_temp = y_collect[idx_x_train].astype(int)
y_test_temp = y_collect[~idx_x_train].astype(int)

#one hot encode
y_train_new = np.reshape(pd.get_dummies(y_train_temp).values,\
                         len(y_train_temp)*10, order = 'F')
y_test_new = np.reshape(pd.get_dummies(y_test_temp).values,\
                         len(y_test_temp)*10, order = 'F')

SE = MultiClassKernel(num_classes = 10, params=[[2.2,2.6]],\
                      base_kernel = SquaredExponentialKernel)
K = SE(x_train_new)

MC_GP = Multiclass_GP(SE, y_train_new, 10, K=K)
f, stats = MC_GP.inference(tol = 1e-6, maxiter = 50)
out = MC_GP.predict(x_test_new, x = x_train_new)

