from src.Data.loaddata import loademotiondata
from src.Data.featureextraction import temporalfeatures

data = loademotiondata('SCCN_data/emotion/ab75')
tempfeat = temporalfeatures(data)