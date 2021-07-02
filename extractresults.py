import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det
from scipy.stats import entropy
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

df_collect = pickle.load(open('features/data_collect.pkl', 'rb'))
val, count = np.unique(df_collect['label'], return_counts=True)
hist = count/np.sum(count)
labs = ['Mixed', 'Blink', 'Lat. eyes', 'Muscle', 'Heart', 'Neural']
x_plac = np.array(range(len(labs)))

fig, ax = plt.subplots(figsize=(10,5))    
ax.bar(labs, hist)#, tick_label = labs)

for i, v in enumerate(hist):
    ax.text(i-0.25, v + 0.02, '{:.2f}%'.format(v*100), color='black',\
            fontsize = 'x-large')
plt.ylim([0, 1])
ax.tick_params(labelsize='x-large')
#plt.show()

summary = pickle.load(open('outputs/training_bal2_new.pkl', 'rb'))

accuracies_GP = np.zeros(len(summary.keys()))
accuracies_LR = np.zeros(len(summary.keys()))
top2accGP = np.zeros(len(summary.keys()))
top2accLR = np.zeros(len(summary.keys()))
bestsig = np.zeros(len(summary.keys()))
logliks = np.zeros((len(summary.keys()), len(summary[0]['logliks'])))

for i in range(len(summary.keys())):
    accuracies_GP[i] = summary[i]['accGP']
    accuracies_LR[i] = summary[i]['accLR']
    bestsig[i] = summary[i]['best_sigma']
    logliks[i,:] = summary[i]['logliks']
    if i == 0:
        likGP = summary[i]['likGP']
        likLR = summary[i]['likLR']
        likGPsubj = likGP
        likLRsubj = likLR
        ytest = summary[i]['y_test']
        predGP = np.argmax(likGP, axis = 1)
        predLR = np.argmax(likLR, axis = 1)
        cov = summary[i]['dist']['Cov']
        covdet = np.array([det(cov[k,:,:]) for k in range(cov.shape[0])])
    else:
        likGPsubj = summary[i]['likGP']
        likLRsubj = summary[i]['likLR']
        likGP = np.append(likGP, likGPsubj, axis = 0)
        likLR = np.append(likLR, likLRsubj, axis = 0)
        ytest = np.append(ytest, summary[i]['y_test'], axis =0)
        predGP = np.append(predGP, np.argmax(summary[i]['likGP'],axis = 1), axis =0)
        predLR = np.append(predLR, np.argmax(summary[i]['likLR'], axis = 1), axis =0)
        cov = summary[i]['dist']['Cov']
        covdet = np.append(covdet, \
                           np.array([det(cov[k,:,:]) for k in range(cov.shape[0])]),\
                           axis = 0)
    top2predGP = np.zeros(len(likLRsubj))
    top2predLR = np.zeros(len(likLRsubj))
    for j in range(len(likLRsubj)):
        # get top 2 predictions
        tempGP = np.flip(np.argsort(likGPsubj[j,:]))
        pred = ((tempGP[0] == summary[i]['y_test'][j]) or (tempGP[1] == summary[i]['y_test'][j])) 
        if pred:
            top2predGP[j] = summary[i]['y_test'][j]
        else:
            top2predGP[j] = tempGP[0]
        
        tempLR  = np.flip(np.argsort(likLRsubj[j,:]))
        pred = ((tempLR[0] == summary[i]['y_test'][j]) or (tempLR[1] == summary[i]['y_test'][j]))
        if pred:
            top2predLR[j] = summary[i]['y_test'][j]
        else:
            top2predLR[j] = tempLR[0]
    top2accGP[i] = balanced_accuracy_score(summary[i]['y_test'], top2predGP)
    top2accLR[i] = balanced_accuracy_score(summary[i]['y_test'], top2predLR)

plt.plot(np.arange(3,5, 0.05), logliks.T)
plt.xlabel('Length scale', fontsize=16)
plt.ylabel('Log posterior likelihood', fontsize=16)

plt.show()

wrongclassGP = np.where(ytest!=predGP)
wrongclassLR = np.where(ytest!=predLR)
correctclassGP = np.where(ytest==predGP)
correctclassLR = np.where(ytest==predLR)

likdistGP = np.max(likGP,axis=1)[:,np.newaxis]-likGP
likdistLR = np.max(likLR,axis=1)[:,np.newaxis]-likLR
mindistGP = np.zeros(len(likdistGP))
mindistLR = np.zeros(len(likdistLR))

top2predGP = np.zeros(len(likdistLR))
top2predLR = np.zeros(len(likdistLR))

for i in range(len(likdistGP)):
    # get minimum distance
    tempGP = likdistGP[i,:]
    tempGP = tempGP[np.where(tempGP>0)]
    if len(tempGP)>0:
        mindistGP[i] = np.min(tempGP)
    else:
        mindistGP[i] =0
    tempLR = likdistLR[i,:]
    tempLR = tempLR[np.where(tempLR>0)]
    mindistLR[i] = np.min(tempLR)

    # get top 2 predictions
    tempGP = np.flip(np.argsort(likGP[i,:]))
    pred = ((tempGP[0] == ytest[i]) or (tempGP[1] == ytest[i])) 
    if pred:
        top2predGP[i] = ytest[i]
    else:
        top2predGP[i] = tempGP[0]
    
    tempLR  = np.flip(np.argsort(likLR[i,:]))
    pred = ((tempLR[0] == ytest[i]) or (tempLR[1] == ytest[i]))
    if pred:
        top2predLR[i] = ytest[i]
    else:
        top2predLR[i] = tempLR[0]
    

plt.hist(mindistGP[wrongclassGP], bins=20)
plt.title('Distance, Wrong class, GP', fontsize = 20)
plt.show()
plt.hist(mindistGP[correctclassGP], bins=20)
plt.title('Distance, Correct class, GP', fontsize = 20)
plt.show()
plt.hist(mindistLR[wrongclassLR], bins =20)
plt.title('Distance, Wrong class, LR', fontsize = 20)
plt.show()
plt.hist(mindistLR[correctclassLR], bins =20)
plt.title('Distance, Correct class, LR', fontsize = 20)
plt.show()

entGP = entropy(likGP, axis = 1)
entLR = entropy(likLR, axis = 1)

wrongentGP = entGP[wrongclassGP]
wrongentLR = entLR[wrongclassLR]
correctentGP = entGP[correctclassGP]
correctentLR = entLR[correctclassLR]

plt.hist(wrongentGP, bins = 20)
plt.title('Entropy, wrong class, GP', fontsize = 20)
plt.show()

plt.hist(correctentGP, bins = 20)
plt.title('Entropy, correct class, GP', fontsize = 20)
plt.show()

plt.hist(wrongentLR, bins = 20)
plt.title('Entropy, wrong class, LR', fontsize = 20)
plt.show()

plt.hist(correctentLR, bins = 20)
plt.title('Entropy, correct class, LR', fontsize = 20)
plt.show()

wrongdet = np.log(covdet[wrongclassGP])
correctdet = np.log(covdet[correctclassGP])
plt.hist(wrongdet, bins = 20)
plt.show()
plt.hist(correctdet, bins = 20)
plt.show()

confGP = confusion_matrix(ytest, predGP, normalize='true')
confLR = confusion_matrix(ytest, predLR, normalize='true')

disp = ConfusionMatrixDisplay(confGP, display_labels = labs)
disp.plot(cmap = 'Greys')
plt.show()