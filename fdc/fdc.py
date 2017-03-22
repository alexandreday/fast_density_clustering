'''
Created on Jan 16, 2017

@author: Alexandre Day

    Purpose:
        Fast density clustering
'''

import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors
import time
from sklearn.model_selection import train_test_split
from numpy.random import random
import sys, os

def main():

    '''
        Example for gaussian mixture
    '''
    from sklearn import datasets
    from special_datasets import gaussian_mixture

    n_true_center = 10
    #X,y=datasets.make_blobs(10000, 2, n_true_center, random_state=1984)
    #np.save("data.txt",X)
    
    #exit()
    X,y = gaussian_mixture(n_sample=10000, n_center = n_true_center, sigma_range = [0.25,0.5,1.25],
                            pop_range = [0.1,0.02,0.1,0.1,0.3,0.1,0.08,0.02,0.08,0.1])
                            #random_state = 0)

    model = FDC(nh_size = 40, noise_threshold=0.3)  
    model.fit(X) # Fitting X -> computing density maps and graphs

    idx_centers = model.idx_centers
    cluster_label = model.cluster_label
    rho = model.rho

    plotting.summary(idx_centers, cluster_label, rho, X, n_true_center=n_true_center, y=y, show=True)

    #print("--> Saving in result.dat with format [idx_centers, cluster_label, rho, n_true_center, X, y, delta]")
    #with open("result.dat", "wb") as f:
    #    pickle.dump([idx_centers, cluster_label, rho, n_true_center, X, y, delta],f)
    
class FDC:

    """ Fast Density Clustering via kernel density modelling 

    Parameters
    ----------

    nh_size : int, optional (default: 40)
        Neighborhood size. This is related to the perplexity (in t-SNE)
        and is an effective scale that defines the number of neighbors of each data point.
        Larger datasets usually require a larger perplexity/nh_size. Consider selecting a value
        between 20 and 100. 

    random_state: int, optional (default: 0)
        Random number for seeding random number generator. By default, the
        method generates the same results. This random is used to seed
        the cross-validation (set partitions) which will in turn affect the bandwitdth value

    verbose: int, optional (default: 1)
        Set to 0 if you don't want to see print to screen.

    bandwidth: str, optional (default: 'auto')
        If you want the bandwidth to be set automatically or want to set it yourself.
        Valid options = {'auto' | 'manual'}

    """

    def __init__(self, nh_size=40, noise_threshold=0.4,
                random_state=0, test_size=0.1, verbose=1, bandwidth=None,
                no_merge=False):

        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.nh_size = nh_size
        self.bandwidth = bandwidth
        self.noise_threshold = noise_threshold
        self.no_merge=no_merge        

    def fit(self,X):
        """ Performs density clustering on given data set

        Parameters
        ----------

        X : array, (n_sample, n_feature)
            Data to cluster. 

        Returns
        ----------
        self
        """
        if self.verbose == 0:
            blockPrint()

        n_sample = X.shape[0]
        print("--> Starting clustering with n=%i samples..." % n_sample)
        start = time.time()

        self.nbrs = NearestNeighbors(n_neighbors = self.nh_size, algorithm='kd_tree').fit(X)
        self.nn_dist, self.nn_list = self.nbrs.kneighbors(X)

        if self.bandwidth:
            bandwidthCV = self.bandwidth
        else:
            print("--> Finding optimal bandwidth ...")
            X_train, X_test, y_train, y_test = train_test_split(X, range(X.shape[0]), test_size=self.test_size, random_state=self.random_state)
            bandwidthCV = find_optimal_bandwidth(self, X, X_train, X_test)

        print("--> Using bandwidth = %.3f" % bandwidthCV)

        print("--> Computing density ...")
        compute_density(self, X, bandwidth=bandwidthCV)

        print("--> Finding centers ...")
        compute_delta(self, X, self.rho)
        
        print("--> Found %i potential centers ..." % self.idx_centers_unmerged.shape[0])

        print("--> Mergin overlapping minimal clusters ...")
        self.check_cluster_stability_fast(X, 0.) # given 

        if self.noise_threshold >= 1e-3 :
            print("--> Iterating merging up to specified noise threshold ...")
            self.check_cluster_stability_fast(X, self.noise_threshold) # merging 'unstable' clusters

        print("--> Done in %.3f s" % (time.time()-start))
        
        enablePrint()

        return self
    
    def check_cluster_stability_fast(self, X, noise_threshold = None): # given 
        if self.verbose == 0:
            blockPrint()

        if noise_threshold is None:
            noise_threshold =  self.noise_threshold

        while True: # iterates untill number of cluster does no change ... 

            self.cluster_label = assign_cluster(self.idx_centers_unmerged, self.nn_delta, self.density_graph) # first approximation of assignments 
            self.idx_centers, n_false_pos = check_cluster_stability(self, X, noise_threshold) 
            self.idx_centers_unmerged = self.idx_centers

            if n_false_pos == 0:
                print("--> Found %i stable centers at noise %.3f ..." % (self.idx_centers.shape[0],noise_threshold))
                break
            else:
                print("\t --> Number of false positive = %i ..."%n_false_pos)
                
        enablePrint()


    def coarse_grain(self, X, noise_threshold_i, noise_threshold_f, dnt, compute_hierarchy = False):
        if self.verbose == 0:
            blockPrint()
        
        print("--> Coarse graining until desired noise threshold ...")

        noise_range = list(np.arange(noise_threshold_i, noise_threshold_f, dnt))
        
        hierarchy = []
        self.max_noise = -1
        n_cluster = 0
        
        for nt in noise_range:
            self.check_cluster_stability_fast(X, noise_threshold = nt)
            if compute_hierarchy is True:
                hierarchy.append({'idx_centers': self.idx_centers, 'cluster_labels': self.cluster_label}) # -> the only required information <- 
                if len(self.idx_centers) != n_cluster:
                    n_cluster = len(self.idx_centers)
                    self.max_noise = nt
    
        if compute_hierarchy is True:
            terminal_cluster = hierarchy[-1]['idx_centers'][0]
            hierarchy.append({'idx_centers': [terminal_cluster], 'cluster_labels' : np.zeros(len(self.cluster_label),dtype=int)})
            noise_range.append(1.5*self.max_noise)
            self.hierarchy = hierarchy
            self.noise_range = noise_range

        enablePrint()
        
def bandwidth_estimate(self, X):
    """
    Purpose:
        Gives a rough estimate of the optimal bandwidth (based on the notion 
        of some effective neigborhood)
    """
    #nbrs = NearestNeighbors(n_neighbors=40, algorithm='kd_tree').fit(X)
    
    #nn_dist,_ = nbrs.kneighbors(X)

    return np.median(self.nn_dist[:,-1]), np.mean(self.nn_dist[:,1])
  
def log_likelihood_test_set(bandwidth, X_train, X_test):
    """
    Purpose:
        Fit the kde model on the training set given some bandwidth and evaluates the log-likelihood of the test set
    """
    kde = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', atol=0.0005, rtol=0.0005,leaf_size=40)
    kde.fit(X_train)
    return -kde.score(X_test)

def find_optimal_bandwidth(self, X, X_train, X_test):
    """
    Purpose:
        Given a training and a test set, finds the optimal bandwidth in a gaussian kernel density model
    """
    from scipy.optimize import fminbound

    hest,hmin=bandwidth_estimate(self, X)
    
    #print("rough bandwidth ",hest,hmin)
    
    args=(X_train,X_test)
    
    # We are trying to find reasonable tight bounds (hmin,1.5*hest) to bracket the error function minima

    h_optimal,score_opt,_,niter=fminbound(log_likelihood_test_set,hmin,1.5*hest,args,maxfun=25,xtol=0.01,full_output=True)
    print("--> Found log-likelihood minima in %i evaluations"%niter)
    
    assert abs(h_optimal-1.5*hest) > 1e-4, "Upper boundary reached for bandwidth"
    assert abs(h_optimal-1.5*hmin) > 1e-4, "Lower boundary reached for bandwidth"

    return h_optimal

def compute_density(self, X, bandwidth=1.0):
    """
    Purpose:
        Given an array of data, computes the local density of every point using kernel density estimation
        
    Return:
        kde.score_samples(X),kde
    """
    self.kde=KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', kernel='gaussian', metric='euclidean', atol=0.000005, rtol=0.00005, breadth_first=True, leaf_size=40)
    self.kde.fit(X)
    self.rho = self.kde.score_samples(X)
    
    return self

def compute_delta(self, X, rho = None):
    """
    Purpose:
        Computes distance to nearest-neighbor with higher density
    Return:
        delta,nn_delta,idx_centers,density_graph

    :delta: distance to n.n. with higher density (within some neighborhood cutoff)
    :nn_delta: index of n.n. with ... 
    :idx_centers: list of points that have the largest density in their neigborhood cutoff
    :density_graph: for every point, list of points are incoming (via the density gradient)

    """
    if rho is None:
        rho = self.rho

    n_sample, n_feature = X.shape

    maxdist = np.linalg.norm([np.max(X[:,i])-np.min(X[:,i]) for i in range(n_feature)])
    
    nn_dist, nn_list = self.nn_dist, self.nn_list
    delta = maxdist*np.ones(n_sample, dtype=np.float)
    nn_delta = np.ones(n_sample, dtype=np.int)
    
    density_graph = [[] for i in range(n_sample)] # store incoming leaves
    
    for i in range(n_sample):
        idx = index_greater(rho[nn_list[i]])
        if idx:
            density_graph[nn_list[i,idx]].append(i)
            nn_delta[i] = nn_list[i,idx]
            delta[i] = nn_dist[i,idx]
        else:
            nn_delta[i]=-1
    
    idx_centers=np.array(range(n_sample))[delta > 0.99*maxdist]
    
    self.delta = delta
    self.nn_delta = nn_delta
    self.idx_centers_unmerged = idx_centers
    self.density_graph = density_graph

    return self

def index_greater(array, prec=1e-8):
    """
    Purpose:
        Function for finding first item in an array that has a value greater than the first element in that array
        If no element is found, returns None
    Precision:
        1e-8
    Return:
        int or None
    """
    item=array[0]
    for idx, val in np.ndenumerate(array):
        if val > (item + prec):
            return idx[0]

def find_NH_tree_search(rho, nn_list, idx, delta, cluster_label, search_size = 20):
    """
    Purpose:
        Function for searching for nearest neighbors within
        some density threshold. 
        NH should be an empty set for the inital function call.
    Return:
        List of points in the neighborhood of point idx
    """

    NH=set(nn_list[idx][1:])  # -- minimal NH scale set by perplexity
    new_leaves=nn_list[idx][1:]
    current_label = cluster_label[idx]
    # ------------------> 
    while True:
        if len(new_leaves) == 0: 
            break
        leaves=new_leaves
        new_leaves=[]

        for leaf in leaves:
            if cluster_label[leaf] == current_label : # search neighbors only if in current cluster 
                nn_leaf = nn_list[leaf][1:search_size]
                for nn in nn_leaf:
                    if (rho[nn] > delta) & (nn not in NH):
                        NH.add(nn)
                        new_leaves.append(nn)

    return np.array(list(NH))

def check_cluster_stability(self, X, threshold): 
    """
    Purpose:
        Given the identified cluster centers, performs a more rigourous
        neighborhood search (based on some noise threshold) for points with higher densities.
        This is vaguely similar to a watershed cuts in image segmentation and basically
        makes sure we haven't identified spurious cluster centers w.r.t to some noise threshold (false positive).
    """

    density_graph = self.density_graph
    nn_delta = self.nn_delta
    delta = self.delta
    rho = self.rho
    nn_list = self.nn_list
    idx_centers = self.idx_centers_unmerged 
    cluster_label = self.cluster_label

    n_false_pos = 0             
    idx_true_centers = []

    for idx in idx_centers:

        rho_center = rho[idx]
        delta_rho = rho_center - threshold
        if threshold < 1e-3: # just check nn_list ...
            NH=nn_list[idx][1:]
        else:
            NH = find_NH_tree_search(rho, nn_list, idx, delta_rho, cluster_label)

        label_centers_nn = np.unique([cluster_label[ni] for ni in NH])
        idx_max = idx_centers[ label_centers_nn[np.argmax(rho[idx_centers[label_centers_nn]])] ]
        rho_current = rho[idx]

        if ( rho_current < rho[idx_max] ) & ( idx != idx_max ) : 

            nn_delta[idx] = idx_max
            delta[idx] = np.linalg.norm(X[idx_max]-X[idx])
            density_graph[idx_max].append(idx)

            n_false_pos+=1
        else:
            idx_true_centers.append(idx)
    return np.array(idx_true_centers,dtype=np.int), n_false_pos

def assign_cluster(idx_centers, nn_delta, density_graph):
    """ 
    Purpose:
        Given the cluster centers and the local gradients (nn_delta) assign to every
        point a cluster label
    """
    
    n_center = idx_centers.shape[0]
    n_sample = nn_delta.shape[0]
    cluster_label = -1*np.ones(n_sample,dtype=np.int) # reinitialized every time.
    
    for c, label in zip(idx_centers, range(n_center) ):
        cluster_label[c] = label
        assign_cluster_deep(density_graph[c], cluster_label, density_graph, label)    
    return cluster_label    

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
