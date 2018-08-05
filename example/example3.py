'''
Created on Feb 1, 2017

@author: Alexandre Day

    Purpose:
        Perform density clustering on gaussian mixture
'''

from fdc import FDC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi
from fdc import plotting
import pickle
import numpy as np

n_true_center = 15

np.random.seed(0)

print("------> Example with %i true cluster centers <-------"%n_true_center)

X, y = make_blobs(50007, 2, n_true_center) # Generating random gaussian mixture
X = StandardScaler().fit_transform(X) # always normalize your data :) 

# set eta=0.0 if you have excellent density profile fit (lots of data say)
model = FDC(eta = 0.01)#, atol=0.0001, rtol=0.0001)

model.fit(X) # performing the clustering
exit()
print("Normalized mutual information = %.4f"%nmi(y, model.cluster_label))
plotting.set_nice_font() # nicer plotting font !
plotting.summary_model(model, ytrue=y, show=True, savefile="result.png")


