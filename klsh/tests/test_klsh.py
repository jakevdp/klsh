import numpy as np
from numpy.testing import assert_

from .. import KernelLSH


def test_linear_klsh(seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(1000, 10)
    Y = rng.randn(1000, 10)

    expected = {16: 0.70,
                32: 0.80,
                64: 0.90,
                128: 0.95}

    def check_klsh(nbits, epsilon=0.5):
        klsh = KernelLSH(nbits=nbits, epsilon=epsilon, random_state=rng).fit(X)
        i_approx = klsh.query(Y, 1)
        i_true = klsh.query_brute(Y, 1)

        # Is this the right way to do this check? I'm not sure...
        match_fraction = (i_approx == i_true).sum() * 1. / i_true.size
        assert_(match_fraction > expected[nbits])

    for nbits in [16, 32, 64, 128]:
        yield check_klsh, nbits
