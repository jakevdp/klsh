import numpy as np
from scipy import ndimage, fftpack, signal

from klsh import KernelLSH
from klsh.kernels import crosscorr_kernel


def create_data(n_samples, n_features, corr_length=4, normalize=True):
    """Create some fake timeseries data"""
    X = np.random.random((n_samples, n_features))
    X = ndimage.gaussian_filter(X, [0, corr_length])
    if normalize:
        norm = np.sqrt((X ** 2).sum(1))
        X /= norm[:, np.newaxis]
    return X

# Create some fake data
np.random.seed(0)
X = create_data(2000, 100)
klsh = KernelLSH(X, nbits=12, kernel=crosscorr_kernel, random_state=0)

# Do some queries
for i, nbrs in enumerate(klsh.query_top_k(X[:5], 5)):
    print("{0} {1}".format(i, nbrs))
