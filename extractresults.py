import pickle
import numpy as np

summary = pickle.load(open('outputs/training_005.pkl', 'rb'))

accuracies_GP = np.zeros(len(summary.keys()))
accuracies_LR = np.zeros(len(summary.keys()))
bestsig = np.zeros(len(summary.keys()))

for i in range(len(summary.keys())):
    accuracies_GP[i] = summary[i]['accGP']
    accuracies_LR[i] = summary[i]['accLR']
    bestsig[i] = summary[i]['best_sigma']