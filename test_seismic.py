"""
Here's what it might look like to use Kernelized LSH on seismic data.

We'll use some fake data for the time being.

Here the hash table is a (potentially) expensive computation done once, and
the querying is a very quick operation that can be done multiple times.

The hash function based on the cross-correlation kernel computation on my
CPU takes on order 0.5*(D/1000) seconds per hash, where D is the length of
the time series.

This means that to compute the hash for N entries will take about
0.5 * N * (D/1000) seconds... for N=10000, D=1000, this is between 1 and 2
hours on a single CPU: note that this is a trivially parallelizable operation,
so it could be scaled relatively easily.

Once the hash table is complete, queries run very quickly because they use
hashes (i.e. O[1]: query time is roughly constant no matter how big the
hash table is). This is what makes LSH amazing: at query time, it doesn't
matter how much data you have!

The query will return approximate nearest neighbors in the parameter space
defined by the cross-correlation kernel.
"""
from __future__ import print_function

import numpy as np
from scipy import ndimage, fftpack, signal

from klsh import KernelLSH
from klsh.kernels import crosscorr_kernel
from klsh.utils import timeit


#------------------------------------------------------------
# We need some fake seismic time series: we'll use this
def create_data(n_samples, n_features, corr_length=4, normalize=True):
    """Create some fake timeseries data"""
    X = np.random.random((n_samples, n_features))
    X = ndimage.gaussian_filter(X, [0, corr_length])
    if normalize:
        norm = np.sqrt((X ** 2).sum(1))
        X /= norm[:, np.newaxis]
    return X

#------------------------------------------------------------
# Now the fun begins:

# Create some fake data
np.random.seed(0)
X = create_data(1000, 100)

# Build the hash table
print("Shape of data: N={0}, D={1}".format(*X.shape))
print("Creating hash table over data (this can be slow, O[N D logD]):")
with timeit():
    klsh = KernelLSH(X, nbits=40, kernel=crosscorr_kernel, random_state=0)
print("")

# Do some queries
print("Querying neighbors of first 100 points "
      "(this is fast; O[D logD] per query):")
with timeit():
    nbrs = klsh.query(X[:100], 5)
print("")
print(nbrs[:10])
