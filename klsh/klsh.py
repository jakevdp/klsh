import numpy as np
from sklearn.metrics import pairwise, pairwise_kernels

from .utils import hamming_cdist, create_rng, packbits_axis
from .hamming_ann import HammingBrute, HammingANN



class HashTable(object):
    def __init__(self):
        self._table = collections.defaultdict(list)
    def __getitem__(self, key):
        return self._table[tuple(key)]
    def __setitem__(self, key, val):
        self._table[tuple(key)] = val


class KernelLSH(object):
    """
    Kernelized Locality Sensitive Hashing

    Parameters
    ----------
    sample : array_like
        An array of size [n_samples, n_features] representing the training data
    nbits : integer (default=10)
        Number of bits in the hash. More bits means fewer matches for any
        given input.
    kernel : string or callable
        The kernel to use. See sklearn.metrics.pairwise
    subspace_size : integer (default=300)
        The amount of data to use when approximating the data distribution in
        the kernel subspace (p in paper). Build-time scales as O[p^3], and
        query time scales as O[p].
    t : integer or None (default=min(subspace_size/4, 30))
        number of random objects to use when choosing kernel-space hyperplanes
    random_state : RandomState object, integer, or None
        seed for random number generator used in creating random projections

    Notes
    -----
    This follows the algorithm in Kulis & Grauman (2009)
    """
    def __init__(self, sample, nbits=10, kernel="linear",
                 subspace_size=300, t=None, epsilon=0.3,
                 random_state=None, kernel_kwds=None):
        self.rng = create_rng(random_state)
        self.epsilon = epsilon

        # set the kernel to be used
        if callable(kernel):
            self._kernelfunc = kernel
        elif kernel in pairwise.PAIRWISE_KERNEL_FUNCTIONS:
            self._kernelfunc = pairwise.PAIRWISE_KERNEL_FUNCTIONS[kernel]
        else:
            raise ValueError("Kernel {0} not recognized".format(kernel))

        if kernel_kwds is None:
            self.kernel_kwds = {}
        else:
            self.kernel_kwds = kernel_kwds

        self.sample_ = np.asarray(sample)
        self.nbits_ = nbits
        self._build_hash_table(subspace_size, t)

    def kernelfunc(self, X, Y):
        """Evaluate the kernel. This can be overloaded by subclasses"""
        return self._kernelfunc(X, Y, **self.kernel_kwds)

    def _kernel_matrix(self, X):
        """Compute the kernel matrix between X and the subspace"""
        K = self.kernelfunc(X, self.subsample_)
        K -= K.mean(1)[:, np.newaxis]
        K -= self.Kmean0_
        K += self.Kmean_
        return K        
        
    def _build_hash_table(self, p, t):
        nbits = self.nbits_
        p = min(p, self.sample_.shape[0])

        if t is None:
            t = min(p // 4, 30)

        # Choose random subsample of p elements
        i = self.rng.choice(self.sample_.shape[0], p, replace=False)
        self.subsample_ = self.sample_[i]

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
        e_s = np.zeros((p, nbits))
        i = np.array([self.rng.choice(p, t) for i in range(nbits)]).T
        e_s[i, np.arange(nbits)] = 1
        self.w_ = np.dot(K_half, e_s)

        self.hash_table_ = self.compute_hash(self.sample_, False)
        #self.hash_index_ = HammingBrute(self.hash_table_)
        self.hash_index_ = HammingANN(self.hash_table_,
                                      epsilon=self.epsilon,
                                      random_state=self.rng)
        
    def compute_hash(self, X, pack=True):
        X = np.atleast_2d(X)
        assert X.ndim == 2
        K = self._kernel_matrix(X)

        bits = (np.dot(K, self.w_) > 0)
        if pack:
            return packbits_axis(bits)
        else:
            return bits

    def query_brute(self, X, k):
        K = self.kernelfunc(X, self.sample_)
        return np.argsort(K, 1)[:, :-k-1:-1]
        
    
    def query(self, X, k, khash=None):
        """Query for all matching hashes

        Parameters
        ----------
        X : array_like
            two-dimensional array of query points
        k : int
            number of approximate neighbors to query
        
        Returns
        -------
        indices : list of lists
            a list of lists of indices of overlapping hashes
        """
        Xbits = self.compute_hash(X, pack=False)
        if khash is None:
            khash = 4 * k

        ind_to_check = self.hash_index_.query(Xbits, khash)
        
        ind = np.zeros((X.shape[0], k), dtype=int)

        for i in range(X.shape[0]):
            kernel = self.kernelfunc(X[i:i+1], self.sample_[ind_to_check[i]])
            ind[i] = ind_to_check[i, np.argsort(kernel[0])[:-k-1:-1]]
        return ind
        
