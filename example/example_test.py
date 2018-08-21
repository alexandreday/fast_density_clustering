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
from matplotlib import pyplot as plt

n_true_center = 15

np.random.seed(0)

print("------> Example with %i true cluster centers <-------"%n_true_center)

X, y = make_blobs(10000, 2, n_true_center) # Generating random gaussian mixture
X = StandardScaler().fit_transform(X) # always normalize your data :) 

# set eta=0.0 if you have excellent density profile fit (lots of data say)
model = FDC(eta = 0.01)#, atol=0.0001, rtol=0.0001)

model.fit(X) # performing the clustering

x = np.linspace(-0.5, 0.6,200)
y = 1.5*x+0.15
X_2 = np.vstack([x,y]).T
xy2 = X_2[65]
b=xy2[0]/1.5+xy2[1]
y2 = -x/1.5+b
#rho = np.exp(model.density_model.evaluate_density(X_2))
#plt.plot(rho)
#plt.show()
#exit()

plt.scatter(x, y, c="green", zorder=2)
plt.scatter(x, y2, c="green", zorder=2)


#plt.scatter(X[:, 0], X[:,1], c= model.rho, cmap="coolwarm")
plt.scatter(X[:, 0],X[:,1], c=model.cluster_label, cmap="jet", alpha=0.1)
plt.xlim([-0.7, 0.7])
plt.ylim([-0.7, 0.7])
plt.show()



""" print("Normalized mutual information = %.4f"%nmi(y, model.cluster_label))
plotting.set_nice_font() # nicer plotting font !
plotting.summary_model(model, ytrue=y, show=True, savefile="result.png")
 """

