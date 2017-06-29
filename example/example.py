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

n_true_center = 15
print("------> Example with %i true cluster centers <-------"%n_true_center)
X, y = make_blobs(10000, 2, n_true_center) # Generating random gaussian mixture

model = FDC(noise_threshold=0.05, nh_size=40) # specifying density clustering parameters

model.fit(X) # performing the clustering

cluster_label, idx_centers, rho = model.cluster_label, model.idx_centers, model.rho # extracting information for plotting

plotting.set_nice_font() # nicer plotting font !

plotting.summary(idx_centers, cluster_label, rho, X, n_true_center, y, savefile="result.png",show=True) # plotting clusters
