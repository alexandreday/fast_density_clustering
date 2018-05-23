from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from fdc import FDC, plotting
import time
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(0)

random_state=170
n_samples = 10000
X,y = make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
X = StandardScaler().fit_transform(X)


model = FDC(kernel='linear')#, eta=0.9)#, test_ratio_size=0.8)
model.fit(X)
y= model.cluster_label
rho = model.rho
print("here:",np.std(np.sort(rho)[1000:]))
print("moment3:\t",np.percentile(rho,75)-np.percentile(rho,25))
print("moment2:\t",np.mean(np.abs(rho-np.mean(rho))))
plt.hist(rho,bins=50)
plt.show()
plotting.cluster_w_label(X,y)
plotting.density_map(X,rho)

plt.show()



