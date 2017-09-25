import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors

class KDE():
    """Kernel density estimation (KDE) for accurate local density estimation.
    This is achieved by using maximum-likelihood estimation of the generative kernel density model
    which is regularized using cross-validation.


    Parameters
    ----------
    bandwidth: float, optional
        bandwidth for the kernel density estimation. If not specified, will be determined automatically using
        maximum likelihood on a test-set.

    nh_size: int, optional
        number of points in a typical neighborhood... only relevant for evaluating
        a crude estimate of the bandwidth. If run in combination with t-SNE, should be on
        the order of the perplexity.

    xtol,atol,rtol: float, optional
        precision parameters for kernel density estimates and bandwidth optimization determination.

    test_ratio_size: float, optional
        ratio of the test size for determining the bandwidth.
    """


    def __init__(self, bandwidth = None, test_ratio_size = 0.1,
                xtol = 0.01, atol=0.000005, rtol=0.00005, extreme_dist = False,
                nn_dist = None):
                
        self.bandwidth = bandwidth
        self.test_ratio_size = test_ratio_size
        self.xtol = xtol
        self.atol = atol
        self.rtol = rtol
        self.extreme_dist = extreme_dist
        self.nn_dist = nn_dist
    
    def fit(self, X):
        """Fit kernel model to X"""
        if self.bandwidth is None:
            self.bandwidth = self.find_optimal_bandwidth(X)
        else:
            self.kde=KernelDensity(
                bandwidth=self.bandwidth, algorithm='kd_tree', 
                kernel='gaussian', metric='euclidean',
                atol=self.atol, rtol=self.rtol, 
                breadth_first=True, leaf_size=40
            )
        
        self.kde.fit(X)

    def evaluate_density(self, X, bandwidth=1.0):
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
        if self.nn_dist is None:
            nn = NearestNeighbors(n_neighbors=2,algorithm='kd_tree')
            nn.fit(X)
            nn_dist, _ = nn.kneighbors(X, n_neighbors=2, return_distance=True)
        else:
            nn_dist = self.nn_dist

        h_min = np.mean(nn_dist[:,1])
        h_max = 5*h_min # heuristic bound !! careful !!

        return h_max, h_min
    
    def find_optimal_bandwidth(self, X):
        """Performs maximum likelihood estimation on a test set of the density model fitted on a training set
        """
        from scipy.optimize import fminbound

        hest, hmin = self.bandwidth_estimate(X)
        print("[kde] Minimum bound = %.4f \t Rough estimate of h = %.4f"%(hmin, hest))

        X_train, X_test = train_test_split(X, test_size = self.test_ratio_size)
        args = (X_train, X_test)

        # We are trying to find reasonable tight bounds (hmin,1.5*hest) to bracket the error function minima
        if self.xtol > hmin:
            tmp = round_float(hmin)
            print('[kde] Bandwidth tolerance (xtol) greater than minimum bound, adjusting xtol: %.5f -> %.5f'%(self.xtol, tmp))
            self.xtol = tmp

        h_optimal, score_opt, _, niter = fminbound(self.log_likelihood_test_set, hmin, 1.5*hest, args, maxfun=100, xtol=self.xtol, full_output=True)
        
        print("[kde] Found log-likelihood minima in %i evaluations, h = %.5f"%(niter, h_optimal))
        
        if self.extreme_dist is False: # in the case of distribution with extreme variances in density, these bounds will fail ...
            assert abs(h_optimal - 1.5*hest) > 1e-4, "Upper boundary reached for bandwidth"
            assert abs(h_optimal - hmin) > 1e-4, "Lower boundary reached for bandwidth"

        return h_optimal

    '''  def find_nh_size(self, X, h_optimal = None, n_estimate = 100):
        """ Given the optimal bandwidth from the CV score, finds the nh_size (using a binary search) which yield h_opt according 
        to the formula np.median(dist_to_nth_neighor) = h_opt
        """
        if h_optimal is None:
            h_optimal = self.bandwidth # should trigger a bug if this is not defined !

        nn = NearestNeighbors(n_neighbors = n_estimate, algorithm='kd_tree').fit(X)
        nn_dist, _ = self.nbrs.kneighbors(X, n_neighbors = 3*n_estimate)
        max_n = 3*n_estimate
        min_n = 0

        n_var = n_estimate
        while True: # performs binary search until convergence !
            h_est = np.median(nn_dist[:,n_var])
            print(n_var,'\t', h_est)
            if h_est > h_optimal:
                max_n = n_var
                change = round(0.5*(max_n - min_n))+min_n
                if change != n_var:
                    n_var = change
                else:
                    break 
            else:
                min_n = n_var
                change = round(0.5*(max_n - min_n))+min_n
                if change != n_var:
                    n_var = change
                else:
                    break
        return n_var 
    '''

    def log_likelihood_test_set(self, bandwidth, X_train, X_test):
        """Fit the kde model on the training set given some bandwidth and evaluates the log-likelihood of the test set
        """
        self.kde = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', atol=self.atol, rtol=self.rtol,leaf_size=40)
        self.kde.fit(X_train) 
        return -self.kde.score(X_test)

def round_float(x):
    """ Rounds a float to it's first significant digit
    """
    a = list(str(x))
    for i, e in enumerate(a):
        if e != '.':
            if e != '0':
                pos = i
                a[i] = '1'
                break
    return float("".join(a[:pos+1]))




