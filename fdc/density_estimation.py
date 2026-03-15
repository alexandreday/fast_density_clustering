import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, NearestNeighbors

try:
    import fdc_rs
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def epanechnikov_kde_from_nn(
    nn_dist: NDArray[np.float64],
    bandwidth: float,
    dim: int,
    n_total: int | None = None,
) -> NDArray[np.float64]:
    """Compute log-density using Epanechnikov kernel from pre-computed k-NN distances.

    Parameters
    ----------
    nn_dist : array, shape (n_query, k)
        Pre-computed distances to k nearest neighbors.
    bandwidth : float
        Kernel bandwidth.
    dim : int
        Dimensionality of the data.
    n_total : int or None
        Total number of points used as normalization denominator.
        If None, uses nn_dist.shape[0] (appropriate when query == training set).

    Returns
    -------
    log_density : array, shape (n_query,)
    """
    # Volume of unit d-ball
    if dim == 1:
        v_d = 2.0
    elif dim == 2:
        v_d = np.pi
    elif dim == 3:
        v_d = 4.0 / 3.0 * np.pi
    else:
        from scipy.special import gamma
        v_d = np.pi ** (dim / 2.0) / gamma(dim / 2.0 + 1)

    # Epanechnikov normalization: c_d = (d+2) / (2 * V_d)
    c_d = (dim + 2) / (2.0 * v_d)

    u = nn_dist / bandwidth
    kernel_vals = np.where(u <= 1.0, c_d * (1.0 - u * u), 0.0)

    if n_total is None:
        n_total = nn_dist.shape[0]
    density = kernel_vals.sum(axis=1) / (n_total * bandwidth ** dim)
    density = np.maximum(density, 1e-300)
    return np.log(density)


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

    def __init__(
        self,
        bandwidth: float | None = None,
        test_ratio_size: float = 0.1,
        xtol: float = 0.01,
        atol: float = 0.000005,
        rtol: float = 0.00005,
        extreme_dist: bool = False,
        nn_dist: NDArray[np.float64] | None = None,
        kernel: str = 'epanechnikov',
        random_state: int | None = None,
    ) -> None:

        self.bandwidth = bandwidth
        self.test_ratio_size = test_ratio_size
        self.xtol = xtol
        self.atol = atol
        self.rtol = rtol
        self.extreme_dist = extreme_dist
        self.nn_dist = nn_dist
        self.kernel = kernel # epanechnikov other option
        self.random_state = random_state
        self._use_knn_kde = False

    @property
    def _can_use_knn_kde(self) -> bool:
        """True when we can use the fast numpy k-NN KDE path."""
        return self.kernel == 'epanechnikov' and self.nn_dist is not None

    def fit(self, X: NDArray[np.float64]) -> 'KDE':
        """Fit kernel model to X"""
        if X.shape[1] > 8 :
            print('Careful, you are trying to do density estimation for data in a D > 8 dimensional space\n ... you are warned !')

        self._dim = X.shape[1]
        self._n_fit = X.shape[0]

        if self.bandwidth is None:
            if self._can_use_knn_kde:
                self.bandwidth = self._find_optimal_bandwidth_knn(X)
                self._use_knn_kde = True
            else:
                self.bandwidth = self.find_optimal_bandwidth(X)
        else:
            if self._can_use_knn_kde:
                self._use_knn_kde = True
            else:
                self.kde = KernelDensity(
                    bandwidth=self.bandwidth, algorithm='kd_tree',
                    kernel=self.kernel, metric='euclidean',
                    atol=self.atol, rtol=self.rtol,
                    breadth_first=True, leaf_size=40
                )
                self.kde.fit(X)
        return self

    def evaluate_density(
        self,
        X: NDArray[np.float64],
        nn_dist: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Given an array of data, computes the local density of every point using kernel density estimation

        Input
        ------
        Data X : array, shape(n_sample,n_feature)
        nn_dist : array, shape(n_sample, k), optional
            Pre-computed k-NN distances for X. When provided and kernel is
            epanechnikov, uses fast numpy path instead of sklearn.

        Return
        ------
        Log of densities for every point: array, shape(n_sample)
        """
        if nn_dist is not None and self._use_knn_kde:
            assert self.bandwidth is not None
            if _HAS_RUST:
                return np.asarray(fdc_rs.epanechnikov_kde(
                    nn_dist, self.bandwidth, X.shape[1], self._n_fit,
                ))
            return epanechnikov_kde_from_nn(
                nn_dist, self.bandwidth, X.shape[1], n_total=self._n_fit,
            )
        return self.kde.score_samples(X)

    def bandwidth_estimate(self, X_train: NDArray[np.float64], X_test: NDArray[np.float64]) -> tuple[float, float, float]:
        """Gives a rough estimate of the optimal bandwidth (based on the notion of some effective neigborhood)

        Return
        ---------
        bandwidth estimate, minimum possible value : tuple, shape(2)
        """
        if self.nn_dist is None:
            nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
            nn.fit(X_train)
            nn_dist, _ = nn.kneighbors(X_test, n_neighbors=2, return_distance=True)
        else:
            nn_dist = self.nn_dist

        dim = X_train.shape[1]

        # Computation of minimum bound
        # This can be computed by taking the limit h -> 0 and making a saddle-point approx.
        mean_nn2_dist = np.mean(nn_dist[:,1]*nn_dist[:,1])
        h_min = np.sqrt(mean_nn2_dist/dim)

        idx_1 = np.random.choice(np.arange(len(X_train)), size=min([1000, len(X_train)]), replace=False)
        idx_2 = np.random.choice(np.arange(len(X_test)), size=min([1000, len(X_test)]), replace=False)

        max_size = min([len(idx_1),len(idx_2)])

        tmp = np.linalg.norm(X_train[idx_1[:max_size]] - X_test[idx_2[:max_size]], axis=1)

        h_max = np.sqrt(np.mean(tmp*tmp)/dim)
        h_est = 10*h_min
        return h_est, h_min, h_max

    def _find_optimal_bandwidth_knn(self, X: NDArray[np.float64]) -> float:
        """Fast bandwidth optimization using numpy k-NN KDE (epanechnikov only)."""
        X_train, X_test = train_test_split(X, test_size=self.test_ratio_size, random_state=self.random_state)

        if _HAS_RUST:
            hest, hmin, hmax = fdc_rs.bandwidth_estimate(
                self.nn_dist if self.nn_dist is not None else np.zeros((0, 0)),
                X_train, X_test,
            )
        else:
            hest, hmin, hmax = self.bandwidth_estimate(X_train, X_test)

        print("[kde] Minimum bound = %.4f \t Rough estimate of h = %.4f \t Maximum bound = %.4f"%(hmin, hest, hmax))
        if _HAS_RUST:
            self.xtol = fdc_rs.round_float(hmin)
        else:
            self.xtol = round_float(hmin)
        print('[kde] Bandwidth tolerance (xtol) set to precision of minimum bound : %.5f '%(self.xtol))

        dim = X_train.shape[1]
        nh_size = self.nn_dist.shape[1] if self.nn_dist is not None else max(min(int(np.sqrt(len(X))), int(15 * np.log10(len(X)))), 10)
        nh_size = min(nh_size, X_train.shape[0])

        # Build k-NN from training set, query test set
        if _HAS_RUST:
            nn_dist_flat, _ = fdc_rs.knn_query_cross(X_train, X_test[:2000], nh_size)
            nn_dist_test = nn_dist_flat.reshape(min(2000, len(X_test)), nh_size)
        else:
            nbrs_train = NearestNeighbors(n_neighbors=nh_size, algorithm='kd_tree').fit(X_train)
            nn_dist_test, _ = nbrs_train.kneighbors(X_test[:2000])

        n_train = X_train.shape[0]

        if _HAS_RUST:
            h_optimal, niter = fdc_rs.find_optimal_bandwidth(
                nn_dist_test, hmin, hmax, dim, n_train, self.xtol,
            )
        else:
            from scipy.optimize import fminbound

            def neg_log_likelihood(bandwidth: float) -> float:
                log_dens = epanechnikov_kde_from_nn(nn_dist_test, bandwidth, dim, n_total=n_train)
                return -float(np.mean(log_dens))

            h_optimal, score_opt, _, niter = fminbound(
                neg_log_likelihood, hmin, hmax * 0.2,
                maxfun=100, xtol=self.xtol, full_output=True,
            )

        print("[kde] Found log-likelihood maximum in %i evaluations, h = %.5f"%(niter, h_optimal))

        if self.extreme_dist is False:
            assert abs(h_optimal - hmax) > 1e-4, "Upper boundary reached for bandwidth"
            assert abs(h_optimal - hmin) > 1e-4, "Lower boundary reached for bandwidth"

        return h_optimal

    def find_optimal_bandwidth(self, X: NDArray[np.float64]) -> float:
        """Performs maximum likelihood estimation on a test set of the density model fitted on a training set
        """
        from scipy.optimize import fminbound
        X_train, X_test = train_test_split(X, test_size=self.test_ratio_size, random_state=self.random_state)

        hest, hmin, hmax = self.bandwidth_estimate(X_train, X_test)

        print("[kde] Minimum bound = %.4f \t Rough estimate of h = %.4f \t Maximum bound = %.4f"%(hmin, hest, hmax))

        # We are trying to find reasonable tight bounds (hmin, 4.0*hest) to bracket the error function minima
        # Would be nice to have some hard accurate bounds
        self.xtol = round_float(hmin)

        print('[kde] Bandwidth tolerance (xtol) set to precision of minimum bound : %.5f '%(self.xtol))

        self.kde = KernelDensity(algorithm='kd_tree', atol=self.atol, rtol=self.rtol,leaf_size=40, kernel=self.kernel)

        self.kde.fit(X_train)

        # hmax is the upper bound, however, heuristically it appears to always be way above the actual bandwidth. hmax*0.2 seems much better but still convservative
        args = (X_test, X_train)
        h_optimal, score_opt, _, niter = fminbound(self.log_likelihood_test_set, hmin, hmax*0.2, args, maxfun=100, xtol=self.xtol, full_output=True)

        print("[kde] Found log-likelihood maximum in %i evaluations, h = %.5f"%(niter, h_optimal))

        if self.extreme_dist is False: # These bounds should always be satisfied ...
            assert abs(h_optimal - hmax) > 1e-4, "Upper boundary reached for bandwidth"
            assert abs(h_optimal - hmin) > 1e-4, "Lower boundary reached for bandwidth"

        return h_optimal

    #@profile
    def log_likelihood_test_set(self, bandwidth: float, X_test: NDArray[np.float64], X_train: NDArray[np.float64]) -> float:
        """Fit the kde model on the training set given some bandwidth and evaluates the negative log-likelihood of the test set
        """
        self.kde.set_params(bandwidth=bandwidth)
        self.kde.fit(X_train)
        return -self.kde.score(X_test[:2000])

def round_float(x: float) -> float:
    """ Rounds a float to it's first significant digit
    """
    if x == 0.0:
        return 0.0
    a = list(str(x))
    for i, e in enumerate(a):
        if e != '.':
            if e != '0':
                pos = i
                a[i] = '1'
                break
    return float("".join(a[:pos+1]))
