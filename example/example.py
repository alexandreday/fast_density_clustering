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

model = FDC(noise_threshold=0.0, nh_size=40) # specifying density clustering parameters

model.fit(X) # performing the clustering

plotting.set_nice_font() # nicer plotting font !

plotting.summary_model(model, ytrue=y, show=True, savefile="result.png")
