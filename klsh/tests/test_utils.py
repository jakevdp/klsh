import numpy as np
from numpy.testing import assert_allclose
from ..utils import (hamming_hashes, packbits_axis,
                     unpackbits_axis, hamming_cdist)


def test_hamming_hashes():
    assert_allclose(list(hamming_hashes(0, 3)),
                    [0, 1, 2, 4, 3, 5, 6, 7])
    assert_allclose(list(hamming_hashes(1, 3)),
                    [1, 0, 3, 5, 2, 4, 7, 6])


def test_hamming_cdist():
    def check_hamming_cdist(N, nbits):
        X = np.random.randint(2, size=(N, nbits))
        x = packbits_axis(X)

        Y = np.random.randint(2, size=(N, nbits))
        y = packbits_axis(Y)

        H1 = hamming_cdist(x, y, use_broadcasting=True)
        H2 = hamming_cdist(x, y, use_broadcasting=False)
        assert_allclose(H1, H2)

    for N in [50, 100, 200]:
        for nbits in [4, 8, 12]:
            yield check_hamming_cdist, N, nbits


def test_packbits_axis():
    X = np.random.randint(2, size=(7, 8, 9, 10))

    def check_round_trip(axis):
        x = packbits_axis(X, axis)
        X2 = unpackbits_axis(x, axis, X.shape[axis])
        assert_allclose(X, X2)

    for axis in range(3):
        yield check_round_trip, axis
