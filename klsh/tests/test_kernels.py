import numpy as np
from numpy.testing import assert_allclose

from ..kernels import (pairwise_correlate, precompute_fft,
                       crosscorr_kernel, crosscorr_similarity)


def test_pairwise_correlate(random_seed=0):
    rng = np.random.RandomState(random_seed)
    X = rng.rand(15, 60)
    Yreal = rng.rand(25, 60)
    Ycomplex = Yreal + 1j * rng.rand(*Yreal.shape)

    def test_correlate(mode, cplx):
        if cplx:
            Y = Ycomplex
        else:
            Y = Yreal

        p1 = pairwise_correlate(X, Y, mode=mode, fast=True)
        p2 = pairwise_correlate(X, Y, mode=mode, fast=False)
        assert_allclose(p1, p2)

    for mode in ['full']:
        for cplx in [True, False]:
            yield test_correlate, mode, cplx


def test_pairwise_precomputed(random_seed=0):
    rng = np.random.RandomState(random_seed)
    X = rng.rand(15, 60)
    Yreal = rng.rand(25, 60)
    Ycomplex = Yreal + 1j * rng.rand(*Yreal.shape)

    def test_correlate(mode, cplx):
        if cplx:
            Y = Ycomplex
        else:
            Y = Yreal

        Xfft, Yfft, fft_info = precompute_fft(X, Y)
        p1 = pairwise_correlate(X, Y, mode=mode)
        p2 = pairwise_correlate(Xfft, Yfft, mode=mode,
                                fft_precomputed=True, fft_info=fft_info)
        assert_allclose(p1, p2)

    for mode in ['full']:
        for cplx in [True, False]:
            yield test_correlate, mode, cplx


def test_crosscorr_kernel_batchsize(random_seed=0):
    rng = np.random.RandomState(random_seed)
    X = rng.rand(100, 128)

    results = [crosscorr_kernel(X, X, batch_size=batch_size)
               for batch_size in [None, 50, 2500, 3000]]

    for r1, r2 in zip(results, results[1:]):
        assert_allclose(r1, r2)


def test_crosscorr_similarity_batchsize(random_seed=0):
    rng = np.random.RandomState(random_seed)
    X = rng.rand(100, 128)

    results = [crosscorr_similarity(X, X, batch_size=batch_size)
               for batch_size in [None, 50, 2500, 3000]]

    for r1, r2 in zip(results, results[1:]):
        assert_allclose(r1, r2)
