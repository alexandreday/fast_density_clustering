'''
Created on Jan 16, 2017

@author: Alexandre Day

    Purpose:
        Code for performing robust density clustering
'''

import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn import datasets
import time
from sklearn.model_selection import train_test_split
from numpy.random import random
import sys, os

def main():

    '''
        Example for gaussian mixture (the number of cluster center can be changed, but
        adjust the parameters accordingly !)
    '''

    n_true_center = 10
    #X,y=datasets.make_blobs(10000,2,n_true_center,random_state=24)

    X,y = gaussian_mixture(n_sample=10000, n_center = n_true_center, sigma_range = [0.25,0.5,1.25],
                            pop_range = [0.1,0.02,0.1,0.1,0.3,0.1,0.08,0.02,0.08,0.1],
                            )#random_state = 8234)

    dcluster = DCluster(NH_size = 40, noise_threshold=0.3)
    cluster_label, idx_centers, rho, delta, kde_tree = dcluster.fit(X)
    plotting.summary(idx_centers, cluster_label, rho, n_true_center, X ,y)
    print("--> Saving in result.dat with format [idx_centers, cluster_label, rho, n_true_center, X, y, delta]")
    with open("result.dat", "wb") as f:
        pickle.dump([idx_centers, cluster_label, rho, n_true_center, X, y, delta],f)

############################################################################################################
############################################################################################################

class DCluster:
    """ Fast two dimensional density clustering via kernel density modelling

    Parameters
    ----------

    NH_size : int, optional (default: 40)
        Neighborhood size. This is related to the perplexity (in t-SNE)
        and is an effective scale that defines the number of neighbors of each data point.
        Larger datasets usually require a larger perplexity/NH_size. Consider selecting a value
        between 20 and 100.

    random_state: int, optional (default: 0)
        Random number for seeding random number generator

    verbose: int, optional (default: 1)
        Set to 0 if you don't want to see print to screen.

    bandwidth: str, optional (default: 'auto')
        If you want the bandwidth to be set automatically or want to set it yourself.
        Valid options = {'auto' | 'manual'}

    bandwidth_value: float, required if bandwidth='manual'

    """

    def __init__(self, NH_size=40, test_size=0.1,
                 random_state=0, verbose=1,
                 noise_threshold=0.4,
                 bandwidth=None):

        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.NH_size = NH_size
        self.bandwidth = bandwidth
        self.delta_rho_threshold=noise_threshold

    def fit(self,X):
        if self.verbose == 0:
            blockPrint()

        n_sample = X.shape[0]
        print("--> Starting clustering with n=%i samples..." % n_sample)
        start = time.time()

        if self.bandwidth:
            bandwidthCV = self.bandwidth
        else:
            print("--> Finding optimal bandwidth ...")
            X_train, X_test, y_train, y_test = train_test_split(X, range(X.shape[0]), test_size=self.test_size, random_state=self.random_state)
            bandwidthCV = find_optimal_bandwidth(X, X_train, X_test)

        print("--> Using bandwidth = %.3f" % bandwidthCV)

        print("--> Computing density ...")
        rho, kde = compute_density(X, bandwidth=bandwidthCV)

        print("--> Finding centers ...")
        delta, nn_delta, idx_centers, density_graph=compute_delta(X,rho,kde.tree_,cutoff=self.NH_size)
        
        print("--> Checking stability ...")

        _, nn_list=kde.tree_.query(list(X), k=20)
        idx_centers = check_cluster_stability(X, density_graph, nn_delta, delta, rho, nn_list, idx_centers, self.delta_rho_threshold)

        print("--> Assigning labels ...")
        cluster_label = assign_cluster(idx_centers, nn_delta, density_graph)
        
        print("--> Done in %.3f s" % (time.time()-start))
        
        print("--> Found %i centers ! ..." % idx_centers.shape[0])

        enablePrint()
        
        return cluster_label, idx_centers, rho, delta, kde.tree_
    
 
def bandwidth_estimate(X):
    """
    Purpose:
        Gives a rough estimate of the optimal bandwidth (based on the notion 
        of some effective neigborhood)
    """
    nbrs = NearestNeighbors(n_neighbors=40,algorithm='kd_tree').fit(X)
    nn_dist,_ = nbrs.kneighbors(X)

    return np.median(nn_dist[:,-1]), np.mean(nn_dist[:,1])
  
def log_likelihood_test_set(bandwidth,X_train,X_test):
    """
    Purpose:
        Fit the kde model on the training set given some bandwidth and evaluates the log-likelihood of the test set
    """
    kde=KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', atol=0.0005, rtol=0.0005,leaf_size=40)
    kde.fit(X_train)
    return -kde.score(X_test)

def find_optimal_bandwidth(X,X_train,X_test):
    """
    Purpose:
        Given a training and a test set, finds the optimal bandwidth in a gaussian kernel density model
    """
    from scipy.optimize import fminbound

    hest,hmin=bandwidth_estimate(X)
    
    #print("rough bandwidth ",hest,hmin)
    
    args=(X_train,X_test)
    
    # We are trying to find reasonable tight bounds (hmin,1.5*hest) to bracket the minima
    
    h_optimal,score_opt,_,niter=fminbound(log_likelihood_test_set,hmin,1.5*hest,args,maxfun=25,xtol=0.01,full_output=True)
    print("--> Found log-likelihood minima in %i evaluations"%niter)

    return h_optimal

def compute_density(X,bandwidth=1.0):
    """
    Purpose:
        Given an array of data, computes the local density of every point using kernel density estimation
        
    Return: array, shape(n_sample,1)

    """
    kde=KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', kernel='gaussian', metric='euclidean', atol=0.000005, rtol=0.00005, breadth_first=True, leaf_size=40)
    kde.fit(X)
    
    return kde.score_samples(X),kde
    
def compute_delta(X,rho,tree,cutoff=40):
    """
    Purpose:
        Computes distance to nearest-neighbor with higher density
    """
    n_sample,n_feature=X.shape
    
    maxdist=np.linalg.norm([np.max(X[:,i])-np.min(X[:,i]) for i in range(n_feature)])
    
    nn_dist,nn_list=tree.query(list(X), k=cutoff)
    delta=maxdist*np.ones(n_sample,dtype=np.float)
    nn_delta=np.ones(n_sample,dtype=np.int)
    
    density_graph=[[] for i in range(n_sample)] # store incoming leaves
    
    for i in range(n_sample):
        idx=index_greater(rho[nn_list[i]])
        if idx is not None:
            density_graph[nn_list[i,idx[0]]].append(i)
            nn_delta[i]=nn_list[i,idx[0]]
            delta[i]=nn_dist[i,idx[0]]
        else:
            nn_delta[i]=-1
    
    idx_centers=np.array(range(n_sample))[delta > 0.99*maxdist]
    
    return delta,nn_delta,idx_centers,density_graph

def index_greater(array):
    """
    Purpose:
        Fast compiled function for finding first item in an array that has a value greater than the first element in that array
        If no element is found, returns None
    """
    item=array[0]
    for idx, val in np.ndenumerate(array):
        if val > item:
            return idx


def assign_cluster_deep(root,cluster_label,density_graph,label):
    """
    Purpose:
        Recursive function for assigning labels for a tree graph.
        Stopping condition is met when the root is empty (i.e. a leaf has been reached)
        
    """
    
    if not root:  # then must be a leaf !
        return
    else:
        for child in root:
            cluster_label[child]=label
            assign_cluster_deep(density_graph[child],cluster_label,density_graph,label)

def assign_cluster(idx_centers,nn_delta,density_graph):
    """ 
    Purpose:
        Given the cluster centers and the local gradients (nn_delta) assign to every
        point a cluster label
    """
    
    n_center=idx_centers.shape[0]
    n_sample=nn_delta.shape[0]
    cluster_label=-1*np.ones(n_sample,dtype=np.int)
    
    for c,label in zip(idx_centers,range(n_center)):
        cluster_label[c]=label
        assign_cluster_deep(density_graph[c],cluster_label,density_graph,label)    
    return cluster_label    

def find_NH_tree_search(rho, nn_list, idx, delta, search_size = 10):
    """
    Purpose:
        Function for searching for nearest neighbors within
        some density threshold. 
        NH should be an empty set for the inital function call.
    """
    NH=set([])
    new_leaves=[idx]

    while True:
        if len(new_leaves) == 0: 
            break

        leaves=new_leaves
        new_leaves=[]
        for leaf in leaves:
            nn_leaf=nn_list[leaf][1:search_size]
            for nn in nn_leaf:
                if (rho[nn] > delta) & (nn not in NH):
                    NH.add(nn)
                    new_leaves.append(nn)

    return np.array(list(NH))

def check_cluster_stability(X, density_graph, nn_delta, delta, rho, nn_list, idx_centers, threshold):
    """
    Purpose:
        Given the identified cluster centers, performs a more rigourous
        neighborhood search (based on some noise threshold) for points with higher densities.
        This is vaguely similar to a watershed cuts in image segmentation and basically
        makes sure we haven't identified spurious cluster centers w.r.t to some noise threshold (false positive).
    """

    n_false_pos=0
    idx_true_centers=[]

    for idx in idx_centers:

        rho_center = rho[idx]
        delta_rho = rho_center - threshold
        
        NH = find_NH_tree_search(rho, nn_list, idx, delta_rho)
        idx_max = NH[np.argmax(rho[NH])]

        if (rho[idx] < rho[idx_max]) & (idx != idx_max):
            nn_delta[idx] = idx_max
            delta[idx] = np.linalg.norm(X[idx_max]-X[idx])
            density_graph[idx_max].append(idx)
            n_false_pos+=1
        else:
            idx_true_centers.append(idx)

    print("--> Number of false positives = %i ..."%n_false_pos)
    return np.array(idx_true_centers,dtype=np.int)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

if __name__=="__main__":

    from matplotlib import pyplot as plt
    import seaborn as sns
    import plotting
    from special_datasets import gaussian_mixture
    import pickle

    main()