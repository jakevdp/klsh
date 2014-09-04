"""
This is a class which uses the methods in [1] to perform a fast approximate
nearest neighbors search in hamming space.

[1] M. Charikar. Similarity Estimation Techniques from Rounding Algorithms.
    ACM Symposium on Theory of Computing, 2002.
"""
import numpy as np
from .utils import create_rng, packbits_axis, unpackbits_axis, hamming_cdist


class HammingBrute(object):
    def __init__(self, X):
        self.X = np.asarray(X, dtype=np.uint8)
        assert self.X.ndim == 2
        N, n_bits = self.X.shape
        self.x = packbits_axis(self.X)

    def query(self, X, k, return_dist=False):
        X = np.asarray(X, dtype=np.uint8)
        assert X.ndim == 2
        packed_bits = packbits_axis(X)
        cdist = hamming_cdist(packed_bits, self.x)
        ind = np.argsort(cdist, 1)[:, :k]
        if return_dist:
            return ind, cdist[np.arange(ind.shape[0])[:, None], ind]
        else:
            return ind
        


class HammingANN(object):
    def __init__(self, X, epsilon=0.5, random_state=None):
        """
        Parameters
        ----------
        X : array_like
           an [n_samples, n_bits] array of hamming features. These will be
           interpreted as zeros and ones.
        """
        self.X = np.asarray(X, dtype=np.uint8)
        assert self.X.ndim == 2
        N, n_bits = self.X.shape
        self.x = packbits_axis(self.X)
        
        self.rng = create_rng(random_state)

        # choose number of permutations based on epsilon
        M = 2 * int(np.ceil(N ** (1. / (1. + epsilon))))

        perm_indices = np.array([np.random.choice(n_bits, n_bits,
                                                  replace=False)
                                 for i in range(M)])

        # packed_bits will be of shape (M, X.shape[0])
        # this is M bit-permutations applied across all the keys
        permutations = self.X[:, perm_indices]
        permutation_bits = packbits_axis(permutations).T

        # Here's where cython would help immensely. We could store just
        # M permutation-bit arrays, and write a custom sort & binary search
        # which will work on these permutations and orderings.
        sort_indices = np.argsort(permutation_bits, 1)
        sorted_permutations = permutation_bits[np.arange(M)[:, None],
                                               sort_indices]
        unsort_indices = np.argsort(sort_indices, 1)

        #----------------- just a sanity check (TODO: REMOVE THIS)
        print("(sanity check...)")
        repacked_bits = sorted_permutations[np.arange(M)[:, np.newaxis],
                                            unsort_indices]
        assert np.all(repacked_bits == permutation_bits)
        #---------------------------------------------------------
        
        self.sort_indices = sort_indices
        self.unsort_indices = unsort_indices
        self.sorted_permutations = sorted_permutations

    def query(self, X, k, return_dist=False):
        """Query a set of distances

        Parameters
        ----------
        X : array_like
           an [n_samples, n_bits] array of hamming features. These will be
           interpreted as zeros and ones.
        """
        print "querying"
        X = np.asarray(X, dtype=np.uint8)
        assert X.ndim == 2
        packed_bits = packbits_axis(X)

        neighbors = np.zeros([X.shape[0], k], dtype=int)

        if return_dist:
            distances = np.zeros_like(neighbors)

        # TODO: MAKE THIS MORE EFFICIENT
        M, N = self.sorted_permutations.shape
        for i, val in enumerate(packed_bits):
            # find ordered index within each random permutation
            perm_indices = [np.searchsorted(self.sorted_permutations[j], val)
                            for j in range(M)]
            perm_indices = np.asarray(perm_indices)

            # get upper/lower indices within each permutation
            ind_uplo = np.clip(np.vstack([perm_indices, perm_indices + 1]),
                               0, N-1)

            # from indices within the sorted permutations, find the
            # indices from the original set of hashes
            ind_to_check = np.unique(self.sort_indices[range(M), ind_uplo])

            # compute hamming distances for these points, and put into results
            dist = hamming_cdist(val, self.x[ind_to_check])
            nearest = np.argsort(dist[0])[:k]
            neighbors[i, :len(nearest)] = ind_to_check[nearest]
            if return_dist:
                distances[i, :len(nearest)] = dist[0, nearest[:k]]

        if return_dist:
            return neighbors, distances
        else:
            return neighbors
