import numpy as np
from numpy.testing import assert_allclose
from ..utils import hamming_hashes, packbits_axis, unpackbits_axis

def test_hamming_hashes():
    assert_allclose(list(hamming_hashes(0, 3)),
                    [0, 1, 2, 4, 3, 5, 6, 7])
    assert_allclose(list(hamming_hashes(1, 3)),
                    [1, 0, 3, 5, 2, 4, 7, 6])


def test_packbits_axis():
    X = np.random.randint(2, size=(4, 5, 6, 7))

    def check_round_trip(axis):
        x = packbits_axis(X, axis)
        X2 = unpackbits_axis(x, axis, X.shape[axis])
        print X.shape, X2.shape
        assert_allclose(X, X2)

    for axis in range(3):
        yield check_round_trip, axis
