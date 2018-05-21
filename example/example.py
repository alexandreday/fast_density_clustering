'''
Created on Feb 1, 2017

@author: Alexandre Day

    Purpose:
        Perform density clustering on gaussian mixture
'''

from fdc import FDC
from sklearn.datasets import make_blobs
from fdc import plotting
import pickle
import numpy as np

n_true_center = 15

np.random.seed(0)

print("------> Example with %i true cluster centers <-------"%n_true_center)

X, y = make_blobs(10000, 2, n_true_center) # Generating random gaussian mixture

model = FDC(eta=0.05, test_ratio_size=0.1)

#@profile
#np.sum(np.arange(0,1000))

model.fit(X) # performing the clustering

plotting.set_nice_font() # nicer plotting font !

plotting.summary_model(model, ytrue=y, show=True, savefile="result.png")
