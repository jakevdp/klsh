import numpy as np
from sklearn.metrics import pairwise, pairwise_kernels

from .utils import hamming_cdist, create_rng, packbits_axis
from .hamming_ann import HammingBrute, HammingANN, HammingBallTree


HAMMING_METHODS = {"brute": HammingBrute,
                   "ann": HammingANN,
                   "balltree": HammingBallTree}


class KernelLSH(object):
    """
    Kernelized Locality Sensitive Hashing

    Parameters
    ----------
    nbits : integer (default=32)
        Number of bits in the hash. More bits means fewer matches for any
        given input. Multiples of 8 are most memory-efficient.
    epsilon : float (default=0.5)
        controls the tradeoff between speed & accuracy.
        O[N^(1/(1+eps))] kernel evaluations are done per query point.
    kernel : string or callable (default="linear")
        The kernel to use. See sklearn.metrics.pairwise
    subspace_size : integer (default=300)
        The amount of data to use when approximating the data distribution in
        the kernel subspace (p in paper). Build-time scales as O[p^3], and
        query time scales as O[p].
    sample_size : integer or None (default=min(subspace_size/4, 30))
        number of random objects to use when choosing kernel-space hyperplanes
    random_state : RandomState object, integer, or None
        seed for random number generator used in creating random projections

    Notes
    -----
    This follows the algorithm in Kulis & Grauman (2009)
    """
    def __init__(self, nbits=32, epsilon=0.5, kernel="linear",
                 subspace_size=300, sample_size=None,
                 index_method="balltree", random_state=None,
                 kernel_kwds=None, index_kwds=None):
        self.random_state = random_state
        self.nbits = nbits
        self.epsilon=epsilon
        self.subspace_size = subspace_size
        self.sample_size = sample_size
        self.kernel_evaluations = 0
        self.fitted = False

        self._check_kernel(kernel, kernel_kwds)
        self._check_index_method(index_method, index_kwds)

    def _check_kernel(self, kernel, kernel_kwds=None):
        self.kernel = kernel
        self.kernel_kwds = kernel_kwds or {}

        if callable(kernel):
            self._kernelfunc = kernel
        elif kernel in pairwise.PAIRWISE_KERNEL_FUNCTIONS:
            self._kernelfunc = pairwise.PAIRWISE_KERNEL_FUNCTIONS[kernel]
        else:
            raise ValueError("Kernel {0} not recognized".format(kernel))

    def _check_index_method(self, index_method, index_kwds=None):
        self.index_method = index_method
        self.index_kwds = index_kwds or {}

        if any(isinstance(index_method, cls)
               for cls in HAMMING_METHODS.values()):
            self._hash_index = index_method
        if index_method in HAMMING_METHODS:
            self._hash_index = HAMMING_METHODS[index_method](**self.index_kwds)
        elif index_method in HAMMING_METHODS.values():
            self._hash_index = index_method(**self.index_kwds)
        else:
            raise ValueError("Unrecognized index_method: "
                             "{0}".format(index_method))

    def fit(self, X):
        """
        Parameters
        ----------
        X : array_like
            An array of size [n_samples, n_features] representing the
            training data.
        """
        # Do these here in case attributes have been modified
        self._check_index_method(self.index_method, self.index_kwds)
        self._check_kernel(self.kernel, self.kernel_kwds)
        self.rng = create_rng(self.random_state)

        self._X_fit = np.asarray(X)
        self._build_hash_table(self.subspace_size, self.sample_size)

        self.fitted = True
        return self

    def kernelfunc(self, X, Y):
        """Evaluate the kernel. This can be overloaded by subclasses"""
        self.kernel_evaluations += X.shape[0] * Y.shape[0]
        return self._kernelfunc(X, Y, **self.kernel_kwds)

    def _kernel_matrix(self, X):
        """Compute the kernel matrix between X and the subspace"""
        K = self.kernelfunc(X, self.subsample_)
        K -= K.mean(1)[:, np.newaxis]
        K -= self.Kmean0_
        K += self.Kmean_
        return K

    def _build_hash_table(self, p, t):
        # p is subspace_size, t is sample_size
        p = min(p, self._X_fit.shape[0])

        if t is None:
            t = min(p // 4, 30)

        # Choose random subsample of p elements
        i = self.rng.choice(self._X_fit.shape[0], p, replace=False)
        self.subsample_ = self._X_fit[i]

        # Compute centered kernel matrix
        K = self.kernelfunc(self.subsample_, self.subsample_)
        self.Kmean0_ = K.mean(0)
        self.Kmean_ = K.mean()
        K -= self.Kmean0_
        K -= self.Kmean0_[:, np.newaxis]
        K += self.Kmean_

        # Compute K^-0.5 via the eigendecomposition
        v, U = np.linalg.eigh(K)
        v[v > 0] = v[v > 0] ** -0.5
        v[v < 0] = 0
        K_half = np.dot(U * v, U.T)

        # Choose the random subsample to find hashing weights
        e_s = np.zeros((p, self.nbits))
        i = np.array([self.rng.choice(p, t) for i in range(self.nbits)]).T
        e_s[i, np.arange(self.nbits)] = 1
        self.w_ = np.dot(K_half, e_s)

        self._hash_table = self.compute_hash(self._X_fit, False)
        self._hash_index.fit(self._hash_table)

    def compute_hash(self, X, pack=False):
        X = np.atleast_2d(X)
        assert X.ndim == 2
        K = self._kernel_matrix(X)

        bits = (np.dot(K, self.w_) > 0).astype(np.uint8)
        if pack:
            return packbits_axis(bits)
        else:
            return bits

    def query_brute(self, X, k, return_similarity=False):
        if not self.fitted:
            raise ValueError("Must call fit() before a query")
        if return_similarity:
            raise NotImplementedError("return_similarity=True")
        K = self.kernelfunc(X, self._X_fit)
        return np.argsort(K, 1)[:, :-k-1:-1]

    def query(self, X, k, khash=None, return_similarity=False):
        """Query for all matching hashes

        Parameters
        ----------
        X : array_like
            two-dimensional array of query points
        k : int
            number of approximate neighbors to query
        khash : int (optional)
            number of hash neighbors to consider. Default value is
            int(N ** (1 / (1 + self.epsilon)))

        Returns
        -------
        indices : list of lists
            a list of lists of indices of overlapping hashes
        """
        if not self.fitted:
            raise ValueError("Must call fit() before a query")
        if return_similarity:
            raise NotImplementedError("return_similarity=True")

        X = np.asarray(X)
        Xbits = self.compute_hash(X)
        if khash is None:
            khash = int(self._X_fit.shape[0] ** (1. / (1 + self.epsilon)))

        # need khash to be more than k
        khash = max(khash, k)

        ind_to_check = self._hash_index.query(Xbits, khash)
        ind = np.zeros((X.shape[0], k), dtype=int)

        for i in range(X.shape[0]):
            kernel = self.kernelfunc(X[i:i+1], self._X_fit[ind_to_check[i]])
            ind[i] = ind_to_check[i, np.argsort(kernel[0])[:-k-1:-1]]
        return ind
