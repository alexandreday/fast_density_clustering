'''
Created on Feb 4, 2017

@author: Alexandre Day

    Perform density clustering on some datasets found in the sklearn 
    documentation on clustering (http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
'''

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from fdc import FDC

np.random.seed(0)

# Generating four datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05) # 
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plot_num = 1

plt.figure(figsize=(10, 10))

"""
Global clustering parameters
"""

noise_threshold=1.0
nh_size=50


datasets = [noisy_circles, noisy_moons, blobs, no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # create clustering estimators

    model=FDC(noise_threshold=noise_threshold, nh_size=nh_size)
    s=time.time()
    model.fit(X)
    cluster_label, idx_centers, rho = model.cluster_label, model.idx_centers, model.rho
    dt=time.time()-s

    n_center=len(idx_centers)

    plt.subplot(2,2,plot_num)
    plt.scatter(X[:, 0], X[:, 1], color=colors[cluster_label].tolist(), s=10,zorder=1)
    plt.text(.99, .01, ('%.2fs' % (dt)).lstrip('0'),
                 transform=plt.gca().transAxes, size=25,
                 horizontalalignment='right',zorder=2)
    plot_num+=1

plt.suptitle("Local density clustering with noise threshold = %.2f \n and neighborhood size = %i. Number of data points = %i"%(noise_threshold,nh_size,n_samples))

plt.savefig("sklearn_datasets.png")
#plt.tight_layout()
plt.show()
