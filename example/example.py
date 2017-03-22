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
X, y = make_blobs(10000, 2, n_true_center)
model = FDC(noise_threshold=0.05, nh_size=40)
model.fit(X)
cluster_label, idx_centers, rho = model.cluster_label, model.idx_centers, model.rho

plotting.summary(idx_centers, cluster_label, rho, X, n_true_center, y, savefile="result.png",show=True)

#print("--> Saving in result.dat with format [idx_centers, cluster_label, rho, n_true_center, X, y, delta]")
#with open("result.dat", "wb") as f:
#    pickle.dump([idx_centers, cluster_label, rho, n_true_center, X, y, delta],f)

