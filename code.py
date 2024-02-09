import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rawdata=pd.read_csv('mnist_train.csv')
idx1=rawdata['label']==1
idx2=rawdata['label']==2

ones=rawdata.iloc[idx1.tolist()]
twos=rawdata.iloc[idx2.tolist()]

ones=ones.drop(['label'], axis=1).to_numpy()
twos=twos.drop(['label'], axis=1).to_numpy()
print(np.shape(ones))
print(np.shape(twos))
