import pandas as pd
import pickle
import numpy as np
from src.GP.gp_algorithms import Multiclass_GP
from src.GP.kernels import SquaredExponentialKernel, MultiClassKernel
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

df_collect = pickle.load(open('features/data_collect.pkl', 'rb'))
subjects = df_collect['subject'].unique()

kfold = KFold(n_splits = 35)
splits = kfold.split(subjects)
sigmas = np.arange(3,5, step = 0.05)

balanced_sampling = True
min_balanced_sampling = True

summary = dict()

j = 0
for (train,test) in splits:
    print('Split', j+1, 'out of 34')
    train_subjects = subjects[train]
    test_subjects = subjects[test]
    # Get train dataset
    df_train = df_collect[np.isin(df_collect['subject'], \
                                    train_subjects)].reset_index()
    X = df_train[df_collect.columns[:-2]].values
    y = df_train['label'].values

    # sample 50% of the mixed components
    if not balanced_sampling:
        mixed = df_train.index[df_train['label']==0]
        n_mixed = (df_train['label']==0).sum()
        n_samp_mixed = int(n_mixed*0.08)
        mixed_sampled = np.random.choice(mixed, n_samp_mixed)
        not_mixed = df_train.index[y!=0]
        X_train = np.append(X[mixed_sampled,:], X[not_mixed,:], axis = 0)
        y_temp = np.append(y[mixed_sampled], y[not_mixed], axis = 0)
    else:
        val, counts = np.unique(y, return_counts = True)
        argmincount = np.argmin(counts)
        mincount = counts[argmincount]
        if min_balanced_sampling:
            samples = mincount
        else:
            samples = int(mincount*1.5)
            
        temp_count = 0
        for lab in val:
            lab_idx = np.where(y==lab)[0]
            if len(lab_idx) < samples:
                samp_lab = np.random.choice(lab_idx, samples, replace = True)
            else:
                samp_lab = np.random.choice(lab_idx, samples, replace = False)

            if temp_count==0:
                X_train = X[samp_lab,:]
                y_temp = y[samp_lab]
            else:
                X_train = np.append(X_train, X[samp_lab,:], axis =0)
                y_temp = np.append(y_temp, y[samp_lab], axis = 0)
            temp_count += 1


    #transform y_train to format used by GP inference
    y_train = np.reshape(pd.get_dummies(y_temp).values,\
                        (6*len(y_temp),), order = 'F')
    
    #Get test dataset
    df_test = df_collect[np.isin(df_collect['subject'], test_subjects)]
    X_test = df_test[df_collect.columns[:-2]].values
    y_test = df_test['label'].values

    # Standardize the train and test set
    X_stand = (X_train-np.mean(X_train, axis = 0)[np.newaxis,:])\
               /np.std(X_train, axis = 0)[np.newaxis,:]
    X_test_stand = (X_test-np.mean(X_train, axis = 0)[np.newaxis,:])\
                    /np.std(X_train, axis = 0)[np.newaxis,:]
    f_init = None
    sol = []
    logliks = np.zeros(len(sigmas))
    i = 0
    # Loop over sigmas
    for sig in sigmas:
        SE = MultiClassKernel(num_classes = 6, params=[[sig,1]],\
                    base_kernel = SquaredExponentialKernel)
        K = SE(X_stand)
        MC_GP = Multiclass_GP(SE, y_train, 6, K=K)
        f, stats = MC_GP.inference(tol = 1e-6, maxiter = 20, f_init = f_init)
        # initialize f at previous solution
        f_init = MC_GP.f
        logliks[i] = stats['lml']
        sol.append(MC_GP)
        i+=1

    #Get test results based on the kernel with maximum likelihood 
    max_loglik = np.argmax(logliks)
    MC_GP = sol[max_loglik]
    sig = sigmas[max_loglik]

    out = MC_GP.predict(X_test_stand, x = X_stand, S=500)
    pred = np.argmax(out[0], axis = 1)

    #Fit with Logistic regression
    X_stand = (X-np.mean(X, axis = 0)[np.newaxis,:])\
               /np.std(X, axis = 0)[np.newaxis,:]
    X_test_stand = (X_test-np.mean(X, axis = 0)[np.newaxis,:])\
                    /np.std(X, axis = 0)[np.newaxis,:]
    LR = LogisticRegression(multi_class= 'multinomial', class_weight='balanced', max_iter = 100)
    LR.fit(X_stand, y)
    out_LR = LR.predict(X_test_stand)
    lik_LR = LR.predict_proba(X_test_stand)

    summary[j] = dict()
    summary[j]['test_split'] = test 
    summary[j]['best_sigma'] = sig
    summary[j]['y_test'] = y_test
    summary[j]['predGP'] = pred
    summary[j]['likGP'] = out[0]
    summary[j]['accGP'] = balanced_accuracy_score(y_test, pred)
    summary[j]['dist'] = out[1]
    summary[j]['predLR'] = out_LR
    summary[j]['accLR'] = balanced_accuracy_score(y_test, out_LR)
    summary[j]['logliks'] = logliks
    summary[j]['likLR'] = lik_LR
    summary[j]['y_train'] = y
    j+=1

pickle.dump(summary, open('outputs/training_sigdelt.pkl', 'wb'))
