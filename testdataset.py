from scipy.io import loadmat

subject1 = loadmat('AlmaEEGdata/AM50127.mat')
subject1_data = subject1['data']['trial'][0][0]$$
subject1_labels = subject1['trialtable']

