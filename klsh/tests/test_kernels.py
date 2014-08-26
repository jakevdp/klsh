import numpy as np
from numpy.testing import assert_allclose

from ..kernels import pairwise_correlate


def test_pairwise_correlate(random_seed=0):
    rng = np.random.RandomState(random_seed)
    X = rng.rand(15, 60)
    Yreal = rng.rand(25, 60)
    Ycomplex = Yreal + 1j * rng.rand(*Yreal.shape)

    def test_correlate(mode, cplx):
        if cplx:
            p1 = pairwise_correlate(X, Ycomplex, mode=mode, fast=True)
            p2 = pairwise_correlate(X, Ycomplex, mode=mode, fast=False)
        else:
            p1 = pairwise_correlate(X, Yreal, mode=mode, fast=True)
            p2 = pairwise_correlate(X, Yreal, mode=mode, fast=False)
        assert_allclose(p1, p2)

    for mode in ['full']:
        for cplx in [True, False]:
            yield test_correlate, mode, cplx
        

