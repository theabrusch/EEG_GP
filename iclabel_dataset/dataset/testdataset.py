import os 
#from icldata import ICLabelDataset
from iclabel_dataset.dataset.icldata import ICLabelDataset
import h5py
import numpy as np

icl = ICLabelDataset(datapath = 'iclabel_dataset/dataset/', label_type='luca')
#icl = ICLabelDataset()
icl_train_data = icl.load_data()

print('hej')