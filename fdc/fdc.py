'''
Created : Jan 16, 2017
Last major update : June 26, 2017

@author: Alexandre Day

    Purpose:
        Fast density clusterin#
'''

import numpy as np
import time
from numpy.random import random
import sys, os
from .density_estimation import KDE
    
class FDC:

    """ Fast Density Clustering via kernel density modelling 

    Parameters
    ----------

    nh_size : int, optional (default: 40)
        Neighborhood size. This is related to the perplexity (in t-SNE)
        and is an effective scale that defines the number of neighbors of each data point.
        Larger datasets usually require a larger perplexity/nh_size. Consider selecting a value
        between 20 and 100.
    
    noise_threshold : float, optional (default : 0.4)
        Used to determined the extended neighborhood of cluster centers. Points
        that have a relative density difference of less than "noise_threshold" and 
        that are density-reachable, are part of the extended neighborhood.

    random_state: int, optional (default: 0)
        Random number for seeding random number generator. By default, the
        method generates the same results. This random is used to seed
        the cross-validation (set partitions) which will in turn affect the bandwitdth value

    test_ratio_size: float, optional (default: 0.1)
        Ratio size of the test set used when performing maximum likehood estimation.

    verbose: int, optional (default: 1)
        Set to 0 if you don't want to see print to screen.

    bandwidth: float, optional (default: None)
        If you want the bandwidth for kernel density to be set automatically or want to set it yourself.
        By default it is set automatically.
    
    no_merge: bool,

    """

    def __init__(self, nh_size=40, noise_threshold=0.4,
                random_state=0, test_ratio_size=0.1, verbose=1, bandwidth=None,
                no_merge=False):

        self.test_ratio_size = test_ratio_size
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
        print("[fdc] Starting clustering with n=%i samples..." % n_sample)
        start = time.time()

        print("[fdc] Fitting kernel model for density estimation ...")
        self.density_model = KDE(bandwidth=self.bandwidth, test_ratio_size=self.test_ratio_size, nh_size=self.nh_size)
        self.density_model.fit(X)

        print("[fdc] Computing density ...")
        self.rho = self.density_model.evalute_density(X)

        print("[fdc] Finding centers ...")
        self.compute_delta(X, self.rho)
        
        print("[fdc] Found %i potential centers ..." % self.idx_centers_unmerged.shape[0])

        print("[fdc] Merging overlapping minimal clusters ...")
        self.check_cluster_stability_fast(X, 0.) # given 

        if self.noise_threshold >= 1e-3 :
            print("[fdc] Iterating merging up to specified noise threshold ...")
            self.check_cluster_stability_fast(X, self.noise_threshold) # merging 'unstable' clusters

        print("[fdc] Done in %.3f s" % (time.time()-start))
        
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
                print("      # of stable clusters with noise %.6f : %i" % (noise_threshold, self.idx_centers.shape[0]))
                break
                
        enablePrint()

    def coarse_grain(self, X, noise_threshold_i, noise_threshold_f, dnt, compute_hierarchy = False):
        """Started from an initial noise scale, progressively merges clusters.
        If specified, saves the cluster assignments at every level of the coarse graining if specified.

        Parameters
        -----------
        compute_hierarchy : bool
            Specifies if hierarchy should be stored (list of cluster assignments at all steps)
            If True, hierarchy is stores in self.hierarchy
        """

        if self.verbose == 0:
            blockPrint()
        
        print("[fdc] Coarse graining until desired noise threshold ...")

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
        

        nn_dist, nn_list = self.density_model.nn_dist, self.density_model.nn_list
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


#####################################################
#####################################################
############ utility functions below ################
#####################################################
#####################################################

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
    Function for searching for nearest neighbors within
    some density threshold. 
    NH should be an empty set for the inital function call.
    
    Returns
    -----------
    List of points in the neighborhood of point idx : 1D array
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
    Given the identified cluster centers, performs a more rigourous
    neighborhood search (based on some noise threshold) for points with higher densities.
    This is vaguely similar to a watershed cuts in image segmentation and basically
    makes sure we haven't identified spurious cluster centers w.r.t to some noise threshold (false positive).
    """

    density_graph = self.density_graph
    nn_delta = self.nn_delta
    delta = self.delta
    rho = self.rho
    nn_list = self.density_model.nn_list
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
    """Blocks printing to screen"""
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """Enables printing to screen"""
    sys.stdout = sys.__stdout__
