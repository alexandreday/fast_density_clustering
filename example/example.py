'''
Created on Feb 1, 2017

@author: Alexandre Day

    Purpose:
        Perform density clustering on gaussian mixture
'''


from clustering.fdc import FDC
from sklearn.datasets import make_blobs
from clustering import plotting
import pickle

n_true_center = 20
print("------> Example with %i true cluster centers <-------"%n_true_center)
X, y = make_blobs(10000, 2, n_true_center)
fdc = FDC(noise_threshold=0.01, NH_size=20)
cluster_label, idx_centers, rho, delta, kde_tree = fdc.fit(X)

plotting.summary(idx_centers, cluster_label, rho, n_true_center, X, y, savefile="result.png")

print("--> Saving in result.dat with format [idx_centers, cluster_label, rho, n_true_center, X, y, delta]")
with open("result.dat", "wb") as f:
    pickle.dump([idx_centers, cluster_label, rho, n_true_center, X, y, delta],f)

