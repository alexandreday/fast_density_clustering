'''
Created : Jan 16, 2017
Last major update : June 29, 2017

@author: Alexandre Day

    Purpose:
        Fast density clustering
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

    nh_size : int, optional (default = 40)
        Neighborhood size. This is the scale used for identifying modes in the density distribution.
        If a point has the maximum density among it's nh_size neighbors, it is marked as 
        a potential cluster center.
    
    noise_threshold : float, optional (default = 0.4)
        Used to merge clusters. This is done by quenching directly to the specified noise threshold
        (as opposed to progressively coarse-graining). The noise threshold determines the extended 
        neighborhood of cluster centers. Points that have a relative density difference of less than 
        "noise_threshold" and that are density-reachable, are part of the extended neighborhood.

    random_state: int, optional (default = 0)
        Random number for seeding random number generator. By default, the
        method generates the same results. This random is used to seed
        the cross-validation (set partitions) which will in turn affect the bandwitdth value

    test_ratio_size: float, optional (default = 0.1)
        Ratio size of the test set used when performing maximum likehood estimation.

    verbose: int, optional (default = 1)
        Set to 0 if you don't want to see print to screen.

    bandwidth: float, optional (default = None)
        If you want the bandwidth for kernel density to be set automatically or want to set it yourself.
        By default it is set automatically.
    
    merge: bool, optinal (default = True)
        Optional merging at zero noise threshold, merges overlapping minimal clusters
    
    atol: float, optional (default = 0.000005)
        kernel density estimate precision parameter. determines the precision used for kde.
        smaller values leads to slower execution but better precision
    
    rtol: float, optional (default = 0.00005)
        kernel density estimate precision parameter. determines the precision used for kde.
        smaller values leads to slower execution but better precision
    
    xtol: float, optional (default = 0.01)
        precision parameter for optimizing the bandwidth using maximum likelihood on a test set
    
    search_size int, optional (default = 20)
        when performing search over neighborhoods, size of each local neighborhood to check when
        expanding. This drastically slows the coarse-graining if chosen to be too big !
    """

    def __init__(self, nh_size=40, noise_threshold=0.4,
                random_state=0, test_ratio_size=0.1, verbose=1, bandwidth=None,
                merge=True,
                atol=0.000005,
                rtol=0.00005,
                xtol=0.01,
                search_size = 20
                ):

        self.test_ratio_size = test_ratio_size
        self.random_state = random_state
        self.verbose = verbose
        self.nh_size = nh_size
        self.bandwidth = bandwidth
        self.noise_threshold = noise_threshold
        self.merge=merge
        self.atol = atol
        self.rtol = rtol
        self.xtol = xtol 
        self.cluster_label = None
        self.search_size = search_size

    def fit(self,X):
        """ Performs density clustering on given data set

        Parameters
        ----------

        X : array, (n_sample, n_feature)
            Data to cluster. 

        Returns
        ----------
        self : fdc object
            To obtain new cluster labels use self.cluster_label
        """
        from sklearn.neighbors import NearestNeighbors

        self.X = X  # shallow copy

        if self.nh_size < 100 :
            self.nbrs = NearestNeighbors(n_neighbors = 100, algorithm='kd_tree').fit(X)
        else:
            self.nbrs = NearestNeighbors(n_neighbors = self.nh_size, algorithm='kd_tree').fit(X)    

        self.nn_dist, self.nn_list = self.nbrs.kneighbors(X) # this scales like X.shape[0] * self.nh_size 

        if self.verbose == 0:
            blockPrint()

        n_sample = X.shape[0]
        print("[fdc] Starting clustering with n=%i samples..." % n_sample)
        start = time.time()

        print("[fdc] Fitting kernel model for density estimation ...")
        self.density_model = KDE(bandwidth=self.bandwidth, test_ratio_size=self.test_ratio_size,
            atol=self.atol,rtol=self.rtol,xtol=self.xtol, nn_dist = self.nn_dist)

        self.density_model.fit(X)
        self.bandwidth = self.density_model.bandwidth

        print("[fdc] Computing density ...")
        self.rho = self.density_model.evaluate_density(X)

        print("[fdc] Finding centers ...")
        self.compute_delta(X, self.rho)
        
        print("[fdc] Found %i potential centers ..." % self.idx_centers_unmerged.shape[0])

        if self.merge: # usually by default one should perform this minimal merging .. 
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

    def coarse_grain(self, noise_iterable):
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

        noise_range = [n for n in noise_iterable]
            
        hierarchy = []
        self.max_noise = -1
        n_cluster = 0
        self.clustering_history = {}

        # note to self, if no merger is done, no need to store hierarchy ... just work with noise_range dict ... 
        
        for nt in noise_range:
            self.check_cluster_stability_fast(self.X, noise_threshold = nt)
            self.clustering_history[round(nt,3)] = (self.cluster_label,self.idx_centers) # storing for later plotting ... 
            
            hierarchy.append({'idx_centers': self.idx_centers, 'cluster_labels': self.cluster_label}) # -> the only required information <- 
            if len(self.idx_centers) != n_cluster:
                n_cluster = len(self.idx_centers)
                self.max_noise = nt

        self.hierarchy = hierarchy
        self.noise_range = noise_range
        self.noise_threshold = noise_range[-1]

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
        delta = maxdist*np.ones(n_sample, dtype=np.float)
        nn_delta = np.ones(n_sample, dtype=np.int)
        
        density_graph = [[] for i in range(n_sample)] # store incoming leaves
        
        ### ----------->
        nn_list = self.nn_list[:,:self.nh_size] # restricted over neighborhood defined by user !
        ### ----------->

        for i in range(n_sample):
            idx = index_greater(rho[nn_list[i]])
            if idx:
                density_graph[nn_list[i,idx]].append(i)
                nn_delta[i] = nn_list[i,idx]
                delta[i] = self.nn_dist[i,idx]
            else:
                nn_delta[i]=-1
        
        idx_centers=np.array(range(n_sample))[delta > 0.99*maxdist]
        
        self.delta = delta
        self.nn_delta = nn_delta
        self.idx_centers_unmerged = idx_centers
        self.density_graph = density_graph

        return self

    def get_cluster_info(self, delta = None):
        """ Returns (cluster_label, idx_center) """

        if delta is None:
            return self.cluster_label, self.idx_centers
        else:
            pos = np.argmin(np.abs(np.array(self.noise_range)-delta))
            #delta_ = self.noise_range[pos]
            #idx_centers = self.hierarchy[pos]['idx_centers']
            cluster_label = self.hierarchy[pos]['cluster_labels']
            idx_center = self.hierarchy[pos]['idx_centers']
            return cluster_label, idx_center

    def update_labels(self, idx_centers, cluster_label):
        self.idx_centers = idx_centers
        self.cluster_label = cluster_label


    def find_NH_tree_search(self, idx, delta, cluster_label):
        """
        Function for searching for nearest neighbors within
        some density threshold. 
        NH should be an empty set for the inital function call.

        Note to myself : lots of optimization, this is pretty time consumming !
        
        Returns
        -----------
        List of points in the neighborhood of point idx : 1D array
        """
        rho = self.rho
        nn_list = self.nn_list

        
        new_leaves=nn_list[idx][1:self.nh_size]
        NH=set(nn_list[idx][1:self.nh_size])  # starts from the initial minimal neighborhood set by user 
        current_label = cluster_label[idx]

        while True:
            if len(new_leaves) == 0: 
                break
            leaves=new_leaves
            new_leaves=[]

            for leaf in leaves:
                if cluster_label[leaf] == current_label : # search neighbors only if in current cluster 
                    nn_leaf = nn_list[leaf][1:self.search_size] # note, this search_size can be greater than self.nh_size !

                    for nn in nn_leaf:
                        if (rho[nn] > delta) & (nn not in NH):
                            NH.add(nn)
                            new_leaves.append(nn)

        return np.array(list(NH))


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
            NH = self.find_NH_tree_search(idx, delta_rho, cluster_label)

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
