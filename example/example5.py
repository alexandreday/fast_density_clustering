import numpy as np
import sklearn.datasets as sd
from fdc import plotting
from sklearn.preprocessing import StandardScaler

np.random.seed(15)
X1, y1 = sd.make_moons(n_samples=500,noise=0.1)
X2, y2 = sd.make_blobs(n_samples=300, centers=2, cluster_std=0.3)
y2+=np.max(y1)+1
X3, y3 = sd.make_blobs(n_samples=1000, centers=3, cluster_std=1.5)
y3+=np.max(y2)+1
X4, y4 = sd.make_blobs(n_samples=1000, centers=3, cluster_std=1)
y4+=np.max(y3)+1
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X4 = np.dot(X4, transformation)
X5, y5 = sd.make_blobs(n_samples=2000, centers=1, cluster_std=4)
X5= X5 - np.array([10,10])
y5+=np.max(y4)+1

X1*=5
X = np.vstack((X1,X2,X3,X4,X5))
y = np.hstack((y1,y2,y3,y4,y5))
#X4 = np.vstack([np.dot(x4,np.array([1,0.5],[0.3,0.5])) for x4 in X4])

from fdc import FDC
plotting.cluster_w_label(X,y)

model = FDC(eta=0.2)
model.fit(StandardScaler().fit_transform(X))


plotting.density_map(X,model.rho,out_file='dmap.pdf')
plotting.cluster_w_label(X,model.cluster_label)

