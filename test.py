import numpy as np
from klsh import KernelLSH
from sklearn.metrics import pairwise

X = np.random.random((100, 10))

klsh = KernelLSH(X, kernel='linear', nbits=5)
for i in range(10):
    nbrs = klsh.query(X[i])
    K = pairwise.rbf_kernel(X[i], X)[0]
    print i, nbrs
    print np.sort(K[nbrs])
    print np.sort(K)[-10:]
    print
