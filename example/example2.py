'''
Created on Feb 4, 2017

@author: Alexandre Day

    Purpose:
        Perform density clustering on some datasets found in
        the sklearn documentation on clustering (http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)
'''

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from clustering.fdc import FDC

np.random.seed(0)

# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

plot_num = 1

plt.figure(figsize=(10, 10))
#plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

####################
###################

noise_threshold=1.0
NH_size=50

##################
#################


datasets = [noisy_circles, noisy_moons, blobs, no_structure]
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # create clustering estimators

    fdc=FDC(noise_threshold=noise_threshold, NH_size=NH_size)
    s=time.time()
    cluster_label, idx_centers, rho, delta, kde_tree = fdc.fit(X)
    dt=time.time()-s

    n_center=len(idx_centers)

    plt.subplot(2,2,plot_num)
    plt.scatter(X[:, 0], X[:, 1], color=colors[cluster_label].tolist(), s=10)
    plt.text(.99, .01, ('%.2fs' % (dt)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
    plot_num+=1

plt.suptitle("Local density clustering with \n noise threshold = %.2f and neighborhood size = %i"%(noise_threshold,NH_size))
#plt.tight_layout()
plt.show()