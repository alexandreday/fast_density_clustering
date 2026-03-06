'''
Created : Jan 16, 2017
Last major update : June 29, 2017

@author: Alexandre Day

    Purpose:
        Fast density clustering
'''


from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray
import time
import sys, os
from .density_estimation import KDE
import pickle
from sklearn.neighbors import NearestNeighbors
import multiprocessing
    
class FDC:

    """ Fast Density Clustering via kernel density modelling for low-dimensional data (D <~ 8)

    Parameters
    ----------

    nh_size : int, optional (default = 'auto')
        Neighborhood size. This is the scale used for identifying the initial modes in the density distribution, regardless
        of the covariance. If a point has the maximum density among it's nh_size neighbors, it is marked as 
        a potential cluster center. 'auto' means that the nh_size is scaled with number of samples. We 
        use nh_size = 100 for 10000 samples. The minimum neighborhood size is set to 10.
    
    eta : float, optional (default = 0.4)
        Noise threshold used to merge clusters. This is done by quenching directly to the specified noise threshold
        (as opposed to progressively coarse-graining). The noise threshold determines the extended 
        neighborhood of cluster centers. Points that have a relative density difference of less than 
        "noise_threshold" and that are density-reachable, are part of the extended neighborhood.

    random_state: int, optional (default = 0)
        Random number for seeding random number generator. By default, the
        method generates the same results. This random is used to seed
        the cross-validation (set partitions) which will in turn affect the bandwitdth value

    test_ratio_size: float, optional (default = 0.8)
        Ratio size of the test set used when performing maximum likehood estimation.
        In order to have smooth density estimations (prevent overfitting), it is recommended to
        use a large test_ratio_size (closer to 1.0) rather than a small one.

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
    
    search_size: int, optional (default = 20)
        when performing search over neighborhoods, size of each local neighborhood to check when
        expanding. This drastically slows the coarse-graining if chosen to be too big !

    kernel: str, optional (default='epanechnikov')
        Type of Kernel to use for density estimates. Other options are {'gaussian'|'linear','tophat'}.
    """

    def __init__(
        self,
        nh_size: int | Literal['auto'] = 'auto',
        eta: float = 0.5,
        random_state: int = 0,
        test_ratio_size: float = 0.8,
        verbose: int = 1,
        bandwidth: float | None = None,
        merge: bool = True,
        atol: float = 0.01,
        rtol: float = 0.0001,
        xtol: float = 0.01,
        search_size: int = 20,
        n_cluster_init: int | None = None,
        kernel: str = 'epanechnikov',
        n_job: int | Literal['auto'] = 'auto',
    ) -> None:

        self.test_ratio_size = test_ratio_size
        self.random_state = random_state
        self.verbose = verbose
        self.nh_size = nh_size
        self.bandwidth = bandwidth
        self.eta = eta
        self.merge = merge
        self.atol = atol
        self.rtol = rtol
        self.xtol = xtol
        self.cluster_label: NDArray[np.int_] | None = None
        self.search_size = search_size
        self.n_cluster_init = n_cluster_init
        self.kernel = kernel
        self.nbrs: NearestNeighbors | None = None
        self.nn_dist: NDArray[np.float64] | None = None
        self.nn_list: NDArray[np.int_] | None = None
        self.density_model: KDE | None = None
        self.X: NDArray[np.float64] | None = None
        self.rho: NDArray[np.float64] | None = None
        self.delta: NDArray[np.float64] | None = None
        self.nn_delta: NDArray[np.int_] | None = None
        self.idx_centers: NDArray[np.int_] | None = None
        self.idx_centers_unmerged: NDArray[np.int_] | None = None
        self.density_graph: list[list[int]] | None = None

        if n_job == 'auto':
            self.n_job = multiprocessing.cpu_count()
        elif n_job > multiprocessing.cpu_count():
            self.n_job = multiprocessing.cpu_count()
        else:
            self.n_job = n_job

    def fit(self, X: NDArray[np.float64]) -> 'FDC':
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
        t = time.time()

        self.X = X  # shallow copy
        self.n_sample = X.shape[0]

        if self.n_sample < 10:
            assert False, "Too few samples for computing densities !"

        if self.nh_size == 'auto':
            # Start with a generous neighborhood for k-NN computation;
            # after bandwidth optimization, it will be adaptively trimmed
            # to neighbors within a few bandwidths.
            self.nh_size = max(int(25*np.log10(self.n_sample)), 10)
        assert isinstance(self.nh_size, int)

        if self.search_size > self.nh_size:
            self.search_size = self.nh_size
        if self.verbose == 0:
            blockPrint()
        
        self.display_main_parameters()

        print("[fdc] Starting clustering with n=%i samples..." % X.shape[0])
        start = time.time()

        print("[fdc] Fitting kernel model for density estimation ...")
        self.fit_density(X)

        # Adaptively restrict neighborhood to neighbors within a few bandwidths.
        # This prevents the neighborhood from bleeding across cluster boundaries
        # when nh_size (set by the auto heuristic) is larger than the cluster size.
        self._adaptive_trim_neighbors()

        print("[fdc] Finding centers ...")
        self.compute_delta(X, self.rho)
        assert self.idx_centers_unmerged is not None
        assert self.nn_delta is not None
        assert self.density_graph is not None

        print("[fdc] Found %i potential centers ..." % self.idx_centers_unmerged.shape[0])

        # temporary idx for the centers :
        self.idx_centers = self.idx_centers_unmerged
        self.cluster_label = assign_cluster(self.idx_centers_unmerged, self.nn_delta, self.density_graph)

        if self.merge: # usually by default one should perform this minimal merging .. 
            print("[fdc] Merging overlapping minimal clusters ...")
            self.check_cluster_stability_fast(X, 0.) # given
            if self.eta >= 1e-3 :
                print("[fdc] Iterating up to specified noise threshold ...")
                self.check_cluster_stability_fast(X, self.eta) # merging 'unstable' clusters
        
        print("[fdc] Done in %.3f s" % (time.time()-start))
        
        enablePrint()

        return self

    def save(self, name: str | None = None) -> str:
        """ Saves current model to specified path 'name' """
        if name is None:
            fname = self.make_file_name()
        else:
            fname = name

        fopen = open(fname,'wb')
        pickle.dump(self,fopen)
        fopen.close()
        return fname
        
    def load(self, name: str | None = None) -> 'FDC':
        if name is None:
            name = self.make_file_name()

        self.__dict__.update(pickle.load(open(name,'rb')).__dict__)
        return self


    def fit_density(self, X: NDArray[np.float64]) -> 'FDC':

        # nearest neighbors class
        self.nbrs = NearestNeighbors(n_neighbors = self.nh_size, algorithm='kd_tree').fit(X)

        # get k-NN
        self.nn_dist, self.nn_list = self.nbrs.kneighbors(X)

        # Deduplicate for KDE fitting/bandwidth optimization.
        # Exact duplicates cause nn_dist[:,1]=0, which makes bandwidth estimation
        # diverge (h_min=0). We fit the KDE on unique points only, then evaluate
        # density on all original points.
        X_unique, unique_idx = np.unique(X, axis=0, return_index=True)
        n_dupes = X.shape[0] - X_unique.shape[0]
        if n_dupes > 0:
            print("[fdc] Found %i duplicate points; using %i unique points for KDE fitting"
                  % (n_dupes, X_unique.shape[0]))
            # nn_dist from unique points only (for bandwidth_estimate)
            nn_dist_unique = self.nn_dist[unique_idx]
        else:
            X_unique = X
            nn_dist_unique = self.nn_dist

        # density model class — fit on unique points to avoid bandwidth divergence
        self.density_model = KDE(bandwidth=self.bandwidth, test_ratio_size=self.test_ratio_size,
            atol=self.atol, rtol=self.rtol, xtol=self.xtol, nn_dist=nn_dist_unique, kernel=self.kernel,
            random_state=self.random_state)

        # fit density model on unique points
        self.density_model.fit(X_unique)

        # save bandwidth
        self.bandwidth = self.density_model.bandwidth

        # compute density map based on kernel density model
        if (self.n_sample > 30000) & (self.n_job !=1) :
            print("[fdc] Computing density with %i threads..."%self.n_job)
            p = multiprocessing.Pool(self.n_job)
            size_split = X.shape[0]//self.n_job
            results =[]

            idx_split = chunkIt(len(X), self.n_job) # find the index to split the array in approx. n_job equal parts.

            for i in range(self.n_job):
                results.append(p.apply_async(self.f_tmp, [X[idx_split[i][0]:idx_split[i][1]], i]))
            results = [res.get() for res in results]  # type: ignore[misc]
            asort = np.argsort([results[i][0] for i in range(self.n_job)])  # type: ignore[index]
            self.rho=np.hstack([results[a][1] for a in asort])

        else:
            print("[fdc] Computing density with 1 thread...")
            self.rho = self.density_model.evaluate_density(X)

        return self        
    
    def f_tmp(self, X_: NDArray[np.float64], i_: int) -> tuple[int, NDArray[np.float64]]:
        """evaluating density and keeping track of threading order"""
        assert self.density_model is not None
        return (i_, self.density_model.evaluate_density(X_))

    def coarse_grain(self, noise_iterable: Iterable[float]) -> 'FDC':
        """Started from an initial noise scale, progressively merges clusters.
        If specified, saves the cluster assignments at every level of the coarse graining.

        Parameters
        -----------
        noise_iterable : iterable of floats
            Should be an iterable containg noise values at which to perform coarse graining. Usually
            one should start from 0 and go to larger values by small increments. The whole clustering
            information is stored in self.hierarchy

        Return
        ------
        self
        
        """
        assert self.X is not None
        assert self.idx_centers is not None
        if self.verbose == 0:
            blockPrint()

        print("[fdc] Coarse graining until desired noise threshold ...")

        noise_range = [n for n in noise_iterable]
            
        self.max_noise: float = -1
        n_cluster = 0

        for nt in noise_range:

            if self.n_cluster_init is not None:
                if len(self.idx_centers) <= self.n_cluster_init:
                    print("[fdc.py]    Reached number of specified clusters [= %i] (or close to), n_cluster = %i"%(self.n_cluster_init,len(self.idx_centers)))
                    break

            self.check_cluster_stability_fast(self.X, eta = nt)
            if len(self.idx_centers) != n_cluster:
                n_cluster = len(self.idx_centers)
                self.max_noise = nt

        self.noise_range = noise_range
        self.noise_threshold = noise_range[-1]

        enablePrint()

        return self 
    
    def _adaptive_trim_neighbors(self, alpha: float = 2.0, min_nh: int = 10) -> None:
        """Restrict nn_list/nn_dist to neighbors within alpha * bandwidth.

        After bandwidth optimization, we know the natural length scale of the
        density.  Neighbors beyond a few bandwidths cannot belong to the same
        local density peak, so including them in compute_delta only causes
        cross-cluster bleeding.

        Parameters
        ----------
        alpha : float
            Multiplier on bandwidth to set the distance cutoff.
        min_nh : int
            Floor on the effective neighborhood size to avoid instability.
        """
        assert self.nn_dist is not None
        assert self.nn_list is not None
        assert self.bandwidth is not None

        cutoff = alpha * self.bandwidth

        # Per-point count of neighbors within cutoff
        within = np.sum(self.nn_dist <= cutoff, axis=1)
        effective_nh = max(int(np.median(within)), min_nh)

        if effective_nh < self.nh_size:
            print("[fdc] Adaptive neighborhood: %i -> %i (cutoff=%.3f*h=%.4f)"
                  % (self.nh_size, effective_nh, alpha, cutoff))
            self.nn_dist = self.nn_dist[:, :effective_nh]
            self.nn_list = self.nn_list[:, :effective_nh]
            self.nh_size = effective_nh

            if self.search_size > self.nh_size:
                self.search_size = self.nh_size

    def compute_delta(self, X: NDArray[np.float64], rho: NDArray[np.float64] | None = None) -> 'FDC':
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
            assert self.rho is not None
            rho = self.rho
        assert self.nn_list is not None
        assert self.nn_dist is not None

        n_sample, n_feature = X.shape

        maxdist = np.linalg.norm([np.max(X[:,i])-np.min(X[:,i]) for i in range(n_feature)])
        delta = maxdist*np.ones(n_sample, dtype=float)
        nn_delta = np.ones(n_sample, dtype=int)

        density_graph: list[list[int]] = [[] for i in range(n_sample)] # store incoming leaves

        ### ----------->
        nn_list = self.nn_list # restricted over neighborhood (nh_size)
        ### ----------->

        for i in range(n_sample):
            idx = index_greater(rho[nn_list[i]])
            if idx:
                density_graph[nn_list[i,idx]].append(i)
                nn_delta[i] = nn_list[i,idx]
                delta[i] = self.nn_dist[i,idx]
            else:
                nn_delta[i]=-1
        
        idx_centers=np.array(range(n_sample))[delta > 0.999*maxdist]
        
        self.delta = delta
        self.nn_delta = nn_delta
        self.idx_centers_unmerged = idx_centers
        self.density_graph = density_graph

        return self

    def estimate_eta(self) -> float:
        """ Based on the density distribution, computes a scale for eta
        Need more experimenting, this is not quite working ...
        """
        assert self.rho is not None
        assert self.nn_delta is not None
        idx = int(self.n_sample/10.)
        idx = np.argsort(self.rho)[:-5*idx]#[2:idx:4*idx]
        drho = []

        for i in idx:
            rho_init = self.rho[i]
            nn_i = self.nn_delta[i]
            while nn_i != -1:
                rho_c = self.rho[nn_i]
                nn_i = self.nn_delta[nn_i]
            drho.append(rho_c- rho_init)

        """ plt.his(drho,bins=60)
        plt.show()
        exit() """
        eta = float(np.mean(drho))#+0.5*np.std(drho)

        self.cout("Using std eta of %.3f"%eta)

        return eta


    def check_cluster_stability_fast(self, X: NDArray[np.float64], eta: float | None = None) -> None:
        assert self.idx_centers_unmerged is not None
        assert self.nn_delta is not None
        assert self.density_graph is not None
        if self.verbose == 0:
            blockPrint()

        if eta is None:
            eta = self.eta

        while True: # iterates untill number of cluster does not change ...

            self.cluster_label = assign_cluster(self.idx_centers_unmerged, self.nn_delta, self.density_graph) # first approximation of assignments
            self.idx_centers, n_false_pos = check_cluster_stability(self, X, eta)
            self.idx_centers_unmerged = self.idx_centers

            if n_false_pos == 0:
                print("      # of stable clusters with noise %.6f : %i" % (eta, self.idx_centers.shape[0]))
                break
                
        enablePrint()

    def find_NH_tree_search(self, idx: int, eta: float, cluster_label: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Function for searching for nearest neighbors within some density threshold.
        NH should be an empty set for the inital function call.
        Note to myself : lots of optimization, this is pretty time/memory consumming !

        Parameters
        -----------
        idx : int
            index of the cluster centroid to start from
        eta : float
            maximum density you can spill over (this is "density_center - eta")
        cluster_label: array of int
            cluster label for every datapoint.

        Returns
        -----------
        List of points in the neighborhood of point idx : 1D array
        """
        assert self.rho is not None
        assert self.nn_list is not None
        rho = self.rho

        zero_array: NDArray[np.bool_] = np.zeros(len(self.nn_list), dtype=bool)

        nn_list = self.nn_list

        zero_array[nn_list[idx, :self.search_size]] = True

        new_leaves = zero_array

        is_NH = (rho > eta) & (new_leaves)

        current_label = cluster_label[idx]

        # This could probably be improved, but at least it's fully vectorized and scalable (NlogN in time and N in memory)

        while True:

            update = False

            leaves=np.copy(new_leaves)

            #y_leave = cluster_label[leaves]

            leaves_cluster = (leaves) & (cluster_label == current_label)

            new_leaves=np.zeros(len(self.nn_list), dtype=bool)

            nn_leaf = np.unique(nn_list[leaves_cluster, :self.search_size].flatten())
            
            res = nn_leaf[is_NH[nn_leaf]==False]
            
            pos = np.where(rho[res] > eta)[0]

            if len(pos) > 0: update=True
            
            is_NH[res[pos]] = True

            new_leaves[res[pos]] = True

            if update is False:
                break

        return np.where(is_NH)[0]

    def find_NH_tree_search_v1(self, idx: int, eta: float, cluster_label: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Function for searching for nearest neighbors within
        some density threshold.
        NH should be an empty set for the inital function call.

        Note to myself : lots of optimization, this is pretty time consumming !

        Returns
        -----------
        List of points in the neighborhood of point idx : 1D array
        """
        assert self.rho is not None
        assert self.nn_list is not None
        rho = self.rho
        nn_list = self.nn_list

        new_leaves=nn_list[idx][:self.search_size]

        is_NH: NDArray[np.int_] = np.zeros(len(self.nn_list), dtype=int)

        is_NH[new_leaves[rho[new_leaves] > eta]] = 1
  
        current_label = cluster_label[idx]

        # ideally here we cythonize what's below... this is highly ineficient ...
        while True:

            update = False
            leaves=np.hstack(new_leaves)
            new_leaves=[]

            y_leave = cluster_label[leaves]
            leaves_cluster = leaves[y_leave == current_label]        
            nn_leaf = nn_list[leaves_cluster]

            for i in range(1, self.search_size):
                res = nn_leaf[is_NH[nn_leaf[:,i]] == 0, i]
                pos = np.where(rho[res] > eta)[0]

                if len(pos) > 0: update=True
                
                is_NH[res[pos]] = 1
                new_leaves.append(res[pos])

            if update is False:
                break

        return np.where(is_NH == 1)[0]

    def cout(self, s: str) -> None:
        print('[fdc] '+s)

    def make_file_name(self) -> str:
        t_name = "fdc_nhSize=%i_eta=%.3f_ratio=%.2f.pkl"
        return t_name%(self.nh_size, self.eta, self.test_ratio_size)

    def display_main_parameters(self) -> None:
        assert isinstance(self.nh_size, int)
        if self.eta != 'auto':
            eta: str = "%.3f" % self.eta
        else:
            eta = str(self.eta)
        out = [
        "[fdc] {0:<20s}{1:<4s}{2:<6d}".format("nh_size",":",self.nh_size),
        "[fdc] {0:<20s}{1:<4s}{2:<6s}".format("eta",":",eta),
        "[fdc] {0:<20s}{1:<4s}{2:<6s}".format("merge",":",str(self.merge)),
        "[fdc] {0:<20s}{1:<4s}{2:<6d}".format("search_size",":",self.search_size),
        "[fdc] {0:<20s}{1:<4s}{2:<6.3f}".format("test_size_ratio",":",self.test_ratio_size)
        ]
        for o in out:
            print(o)

    def reset(self) -> None:
        self.bandwidth = None

#####################################################
#####################################################
############ utility functions below ################
#####################################################
#####################################################

def check_cluster_stability(self: FDC, X: NDArray[np.float64], threshold: float) -> tuple[NDArray[np.int_], int]:
    """
    Given the identified cluster centers, performs a more rigourous
    neighborhood search (based on some noise threshold) for points with higher densities.
    This is vaguely similar to a watershed cuts in image segmentation and basically
    makes sure we haven't identified spurious cluster centers w.r.t to some noise threshold (false positive).

    This has bad memory complexity, needs improvement if we want to run on N>10^5 data points.
    """

    assert self.density_graph is not None
    assert self.nn_delta is not None
    assert self.delta is not None
    assert self.rho is not None
    assert self.nn_list is not None
    assert self.idx_centers_unmerged is not None
    assert self.cluster_label is not None
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
            NH=nn_list[idx][1:self.search_size]
        else:
            NH = self.find_NH_tree_search(idx, delta_rho, cluster_label)

        label_centers_nn = np.unique(self.cluster_label[NH])#[cluster_label[ni] for ni in NH])
        idx_max = idx_centers[ label_centers_nn[np.argmax(rho[idx_centers[label_centers_nn]])] ]
        rho_current = rho[idx]

        if ( rho_current < rho[idx_max] ) & ( idx != idx_max ) : 

            nn_delta[idx] = idx_max
            delta[idx] = np.linalg.norm(X[idx_max]-X[idx])
            density_graph[idx_max].append(idx)

            n_false_pos+=1
        else:
            idx_true_centers.append(idx)

        
    return np.array(idx_true_centers,dtype=int), n_false_pos

def assign_cluster(idx_centers: NDArray[np.int_], nn_delta: NDArray[np.int_], density_graph: list[list[int]]) -> NDArray[np.int_]:
    """ 
    Given the cluster centers and the local gradients (nn_delta) assign to every
    point a cluster label
    """
    
    n_center = idx_centers.shape[0]
    n_sample = nn_delta.shape[0]
    cluster_label = -1*np.ones(n_sample,dtype=int) # reinitialized every time.
    
    for c, label in zip(idx_centers, range(n_center) ):
        cluster_label[c] = label
        assign_cluster_deep(density_graph[c], cluster_label, density_graph, label)    
    return cluster_label    

def assign_cluster_deep(root: list[int], cluster_label: NDArray[np.int_], density_graph: list[list[int]], label: int) -> None:
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
        
def index_greater(array: NDArray[np.float64], prec: float = 1e-8) -> int | None:
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
    for idx, val in np.ndenumerate(array): # slow ! : could be cythonized
        if val > (item + prec):
            return int(idx[0])
    return None

def blockPrint() -> None:
    """Blocks printing to screen"""
    sys.stdout = open(os.devnull, 'w')

def enablePrint() -> None:
    """Enables printing to screen"""
    sys.stdout = sys.__stdout__

def chunkIt(length_seq: int, num: int) -> list[list[int]]:
    avg = length_seq / float(num)
    last = 0.0
    idx_list: list[list[int]] = []

    while last < length_seq:
        idx_list.append([int(last),int(last + avg)])
        last += avg
    
    if len(idx_list) > num:
        idx_list.pop()
        idx_list[-1] = [idx_list[-1][0], length_seq]

    return idx_list
