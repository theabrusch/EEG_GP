import pandas as pd
import pickle
import numpy as np
from src.GP.gp_algorithms import Multiclass_GP
from src.GP.kernels import SquaredExponentialKernel, MultiClassKernel
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.io import loadmat
import matplotlib.pyplot as plt

pickle_paths = glob.glob("features/emotion/*")
df_collect = pd.DataFrame()
labs = ['blink', 'lateyes', 'muscle', 'heart', 'gdcomps']
for path in pickle_paths:
    df_temp = pickle.load(open(path, 'rb'))
    subj = path.split('/')[-1]
    subjpath = 'SCCN_data/emotion/' + subj + '/sources.set'
    file = loadmat(subjpath)['EEG'][0][0]
    labels = np.zeros(len(df_temp))
    i = 1
    for lab in labs:
        try:
            labels[file[lab]] = i
        except:
            print('No', lab, 'comp for', subj)
        i+=1
    df_temp['label'] = labels.astype(np.int32)
    df_collect = df_collect.append(df_temp)

subjects = df_collect['subject'].unique()

kfold = KFold(n_splits = 10)
splits = kfold.split(subjects)
#sigmas = np.arange(1, 3, step = 0.1)
sigmas = [1.7]

summary = dict()

j = 0
for (train,test) in splits:
    train_subjects = subjects[train]
    test_subjects = subjects[test]
    df_train = df_collect[np.isin(df_collect['subject'], \
                                    train_subjects)].reset_index()
    X = df_train[df_collect.columns[:-2]].values
    y = df_train['label'].values
    # sample 50% of the mixed components
    mixed = df_train.index[df_train['label']==0]
    n_mixed = (df_train['label']==0).sum()
    n_samp_mixed = int(n_mixed*0.3)
    mixed_sampled = np.random.choice(mixed, n_samp_mixed)
    not_mixed = df_train.index[y!=0]
    X_train = np.append(X[mixed_sampled,:], X[not_mixed,:], axis = 0)
    y_temp = np.append(y[mixed_sampled], y[not_mixed], axis = 0)
    y_train = np.reshape(pd.get_dummies(y_temp).values,\
                        (6*len(y_temp),), order = 'F')
    df_test = df_collect[np.isin(df_collect['subject'], test_subjects)]
    X_test = df_test[df_collect.columns[:-2]].values
    y_test = df_test['label'].values

    X_stand = (X-np.mean(X, axis = 0)[np.newaxis,:])\
               /np.std(X, axis = 0)[np.newaxis,:]
    
    X_test_stand = (X_test-np.mean(X, axis = 0)[np.newaxis,:])\
                    /np.std(X_test, axis = 0)[np.newaxis,:]
    logliks = np.zeros(len(sigmas))
    i = 0
    for sig in sigmas:
        SE = MultiClassKernel(num_classes = 6, params=[[sig,1]],\
                      base_kernel = SquaredExponentialKernel)
        K = SE(X_train)
        MC_GP = Multiclass_GP(SE, y_train, 6, K=K)
        f, stats = MC_GP.inference(tol = 1e-6, maxiter = 50)
        logliks[i] = stats['lml']
    
    max_loglik = np.argmax(logliks)
    sig = sigmas[max_loglik]
    SE = MultiClassKernel(num_classes = 6, params=[[sig,1]],\
                          base_kernel = SquaredExponentialKernel)
    K = SE(X_train)
    MC_GP = Multiclass_GP(SE, y_train, 6, K=K)
    f, stats = MC_GP.inference(tol = 1e-6, maxiter = 100)
    summary[j] = dict()
    summary[j]['test'] = test 
    summary[j]['best_sigma'] = sig
    summary[j]['y_test'] = y_test
    out = MC_GP.predict(X_test, x = X_stand)
    pred = np.argmax(out[0], axis = 1)
    summary[j]['pred'] = pred
    summary[j]['dist'] = out[1]
    j+=1


plt.matshow(K)
plt.show()