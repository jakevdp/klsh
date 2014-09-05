"""
This is a set of classes to perform fast (approximate) nearest neighbors
searches over Hamming spaces.

[1] M. Charikar. Similarity Estimation Techniques from Rounding Algorithms.
    ACM Symposium on Theory of Computing, 2002.
"""
__all__ = ["HammingANN", "HammingBrute", "HammingBallTree"]

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import BallTree
from .utils import create_rng, packbits_axis, unpackbits_axis, hamming_cdist


class HammingSearchBase(object):
    """Base class for Hamming neighbors search"""
    def fit(self, X):
        raise NotImplementedError('HammingSearchBase.fit')

    def query(self, X, k, return_dist=False):
        raise NotImplementedError('HammingSearchBase.query')

    @staticmethod
    def _validate_input(X, return_compact=True):
        X = np.atleast_2d(np.asarray(X, dtype=np.uint8))
        if X.ndim != 2:
            raise ValueError("Input hamming array must be two dimensions")
        if return_compact:
            return packbits_axis(X)
        else:
            X[X != 0] = 1
            return X


class HammingBrute(HammingSearchBase):
    def __init__(self, compact=False):
        self.compact = compact

    def fit(self, X):
        """Fit a set of hamming vectors

        Parameters
        ----------
        X : array_like
            an array of size (n_features, n_bits). Nonzero entries will be
            evaluated as 1, and zero entries as 0
        """
        if self.compact:
            self._fit_X = self._validate_input(X)
        else:
            self._fit_X = self._validate_input(X, False)
        return self

    def query(self, X, k, return_dist=False):
        if self.compact:
            X = self._validate_input(X)
            cdist = hamming_cdist(X, self._fit_X)
        else:
            X = self._validate_input(X, False)
            cdist = distance.cdist(X, self._fit_X, 'hamming')
        ind = np.argsort(cdist, 1)[:, :k]
        if return_dist:
            rows = np.arange(ind.shape[0])[:, np.newaxis]
            dist = cdist[rows, ind]
            if not self.compact:
                dist = (dist * X.shape[1]).astype(int)
            return ind, dist
        else:
            return ind


class HammingBallTree(HammingSearchBase):
    def __init__(self, leaf_size=40, query_kwds=None):
        self.leaf_size = leaf_size
        self.query_kwds = query_kwds or {}

    def fit(self, X):
        X = self._validate_input(X, return_compact=False)
        self._tree = BallTree(X, metric='hamming', leaf_size=self.leaf_size)
        return self

    def query(self, X, k, return_dist=False):
        X = self._validate_input(X, return_compact=False)

        if return_dist:
            dist, ind = self._tree.query(X, k, return_distance=True)
            return ind, (dist * X.shape[1]).astype(int)
        else:
            return self._tree.query(X, k, return_distance=False)


class HammingANN(HammingSearchBase):
    def __init__(self, epsilon=0.5, random_state=None):
        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X):
        """Fit a set of hamming vectors

        Parameters
        ----------
        X : array_like
            an array of size (n_features, n_bits). Nonzero entries will be
            evaluated as 1, and zero entries as 0
        """
        self._X_fit = self._validate_input(X, False)
        self._X_fit_compact = packbits_axis(self._X_fit)
        N, n_bits = self._X_fit.shape

        # choose number of permutations based on epsilon
        M = 2 * int(np.ceil(N ** (1. / (1. + self.epsilon))))

        rng = create_rng(self.random_state)
        P_indices = np.array([rng.choice(n_bits, n_bits, replace=False)
                              for i in range(M)])

        # P_compact will be of shape (M, X.shape[0]), and contains
        # M bit-permutations applied across all the keys
        P = self._X_fit[:, P_indices]
        P_compact = packbits_axis(P).T

        # Do a lexicographic sort of all the permuted bits.
        # Here's where cython would help immensely. We could store just
        # M permutation-bit arrays, and write a custom sort & binary search
        # which will work on these permutations and orderings.
        sort_indices = np.argsort(P_compact, 1)
        P_compact_sorted = P_compact[np.arange(M)[:, None], sort_indices]
        unsort_indices = np.argsort(sort_indices, 1)

        #----------------- just a sanity check (TODO: REMOVE THIS)
        reordered = P_compact_sorted[np.arange(M)[:, np.newaxis],
                                     unsort_indices]
        assert np.all(reordered == P_compact)
        #---------------------------------------------------------

        self._sort_indices = sort_indices
        self._unsort_indices = unsort_indices
        self._P_compact_sorted = P_compact_sorted
        return self

    def query(self, X, k, return_dist=False):
        """Query a set of distances

        Parameters
        ----------
        X : array_like
           an [n_samples, n_bits] array of hamming features. These will be
           interpreted as zeros and ones.
        """
        X_compact = self._validate_input(X)

        nbrs = np.zeros([X_compact.shape[0], k], dtype=int)

        if return_dist:
            dist = np.zeros_like(nbrs)

        M, N = self._P_compact_sorted.shape

        # TODO: MAKE THIS MORE EFFICIENT
        for i, val in enumerate(X_compact):
            # find ordered index within each random permutation
            P_indices = np.array([np.searchsorted(self._P_compact_sorted[j],
                                                  val) for j in range(M)])

            # get upper/lower indices within each permutation
            ind_uplo = np.clip(np.vstack([P_indices, P_indices + 1]), 0, N-1)

            # from indices within the sorted permutations, find the
            # unique set of indices from the original set of hashes
            ind_to_check = np.unique(self._sort_indices[range(M), ind_uplo])

            # compute hamming distances for these points, and put into results
            distances = hamming_cdist(val, self._X_fit_compact[ind_to_check])
            nearest = np.argsort(distances[0])[:k]
            nbrs[i, :len(nearest)] = ind_to_check[nearest]
            if return_dist:
                dist[i, :len(nearest)] = distances[0, nearest[:k]]

        if return_dist:
            return nbrs, dist
        else:
            return nbrs
