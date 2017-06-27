import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors

class KDE():
    """Kernel density estimation (KDE) for accurate local density estimation.
    This is achieved by using maximum-likelihood estimation of the generative kernel density model
    which is regularized using cross-validation.
    """
    def __init__(self, bandwidth = None, test_ratio_size = 0.1, nh_size = 40):
        self.bandwidth = bandwidth
        self.test_ratio_size = test_ratio_size
        self.nh_size = nh_size
    
    def fit(self, X):
        """Fit kernel model to X"""
        if self.bandwidth is None:
            self.nbrs = NearestNeighbors(n_neighbors = self.nh_size, algorithm='kd_tree').fit(X)
            self.nn_dist, self.nn_list = self.nbrs.kneighbors(X)
            self.bandwidth = self.find_optimal_bandwidth(X)
        else:
            self.kde=KernelDensity(bandwidth=self.bandwidth, algorithm='kd_tree', kernel='gaussian', metric='euclidean', atol=0.000005, rtol=0.00005, breadth_first=True, leaf_size=40)
        
        self.kde.fit(X)

    def evalute_density(self, X, bandwidth=1.0):
        """Given an array of data, computes the local density of every point using kernel density estimation

        Input
        ------
        Data X : array, shape(n_sample,n_feature)

        Return
        ------
        Log of densities for every point: array, shape(n_sample)
        Return:
            kde.score_samples(X)
        """
        return self.kde.score_samples(X)
    
    def bandwidth_estimate(self, X):
        """Gives a rough estimate of the optimal bandwidth (based on the notion of some effective neigborhood)
        
        Return
        ---------
        bandwidth estimate, minimum possible value : tuple, shape(2)
        """
        return np.median(self.nn_dist[:,-1]), np.mean(self.nn_dist[:,1])
    
    def find_optimal_bandwidth(self, X):
        """Performs maximum likelihood estimation on a test set of the density model fitted on a training set
        """
        from scipy.optimize import fminbound

        hest, hmin = self.bandwidth_estimate(X)

        X_train, X_test = train_test_split(X, test_size = self.test_ratio_size)
        args = (X_train, X_test)

        # We are trying to find reasonable tight bounds (hmin,1.5*hest) to bracket the error function minima

        h_optimal, score_opt, _, niter = fminbound(self.log_likelihood_test_set, hmin, 1.5*hest, args, maxfun=25, xtol=0.01, full_output=True)
        
        print("      --> Found log-likelihood minima in %i evaluations"%niter)
        
        assert abs(h_optimal - 1.5*hest) > 1e-4, "Upper boundary reached for bandwidth"
        assert abs(h_optimal - hmin) > 1e-4, "Lower boundary reached for bandwidth"

        return h_optimal

    def log_likelihood_test_set(self, bandwidth, X_train, X_test):
        """Fit the kde model on the training set given some bandwidth and evaluates the log-likelihood of the test set
        """
        self.kde = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', atol=0.0005, rtol=0.0005,leaf_size=40)
        self.kde.fit(X_train) 
        return -self.kde.score(X_test)