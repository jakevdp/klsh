import numpy as np
from numpy.testing import assert_allclose, assert_

from ..hamming_ann import HammingANN, HammingBrute, HammingBallTree


def test_Hamming_Brute(seed=0):
    """Test that brute force methods agree"""
    rng = np.random.RandomState(seed)
    X = rng.randint(2, size=(200, 8))
    Y = rng.randint(2, size=(100, 8))

    h1 = HammingBrute(compact=False).fit(X)
    h2 = HammingBrute(compact=True).fit(X)

    ind1, dist1 = h1.query(Y, 10, return_dist=True)
    ind2, dist2 = h2.query(Y, 10, return_dist=True)

    assert_allclose(dist1, dist2)


def test_Hamming_BallTree(seed=0):
    """Test that the BallTree and Brute methods agree"""
    rng = np.random.RandomState(seed)
    X = rng.randint(2, size=(200, 8))
    Y = rng.randint(2, size=(100, 8))

    h1 = HammingBrute().fit(X)
    h2 = HammingBallTree().fit(X)

    ind1, dist1 = h1.query(Y, 10, return_dist=True)
    ind2, dist2 = h2.query(Y, 10, return_dist=True)

    assert_allclose(dist1, dist2)


def test_Hamming_ANN(seed=0):
    """Test that the Approximate Hamming search works"""
    rng = np.random.RandomState(seed)
    X = rng.randint(2, size=(200, 16))
    Y = rng.randint(2, size=(100, 16))

    h1 = HammingBrute().fit(X)
    ind1, dist1 = h1.query(Y, 10, return_dist=True)

    def check_hamming_ann(epsilon):
        h2 = HammingANN(epsilon=epsilon, random_state=rng).fit(X)
        ind2, dist2 = h2.query(Y, 10, return_dist=True)

        # This comparison works... sort of.
        # We should really be testing sets of distances, but I'm not sure...
        assert_(np.sum(dist1 == dist2) > (0.9 - epsilon) * dist1.size)

    for epsilon in (0.0, 0.1, 0.2, 0.3):
        yield check_hamming_ann, epsilon
