import numpy as np
from scipy import fftpack, signal

__all__ = ["crosscorr_kernel", "crosscorr_similarity"]


def pairwise_correlate_slow(X, Y, mode='full'):
    X, Y = map(np.atleast_2d, (X, Y))
    assert X.ndim == 2
    assert Y.ndim == 2

    Y = Y[:, ::-1]

    first_result = signal.fftconvolve(X[0], Y[0], mode)
    M = np.zeros((X.shape[0], Y.shape[0], len(first_result)),
                 dtype=first_result.dtype)

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            M[i, j] = signal.fftconvolve(X[i], Y[j], mode)
    return M


def precompute_fft(X, Y):
    """Pre-compute the FFT of X and Y for use in pairwise_correlate"""
    X, Y = map(np.atleast_2d, (X, Y))
    assert X.ndim == 2
    assert Y.ndim == 2

    Y = Y[:, ::-1]

    s1 = X.shape[1]
    s2 = Y.shape[1]
    size = s1 + s2 - 1
    complex_result = (np.issubdtype(X.dtype, np.complex) or
                      np.issubdtype(Y.dtype, np.complex))

    # Always use 2**n-sized FFT
    fsize = [int(2 ** np.ceil(np.log2(size)))]

    if not complex_result:
        X_fft = np.fft.rfftn(X, fsize)
        Y_fft = np.fft.rfftn(Y, fsize)
    else:
        X_fft = np.fft.fftn(X, fsize)
        Y_fft = np.fft.fftn(Y, fsize)

    return X_fft, Y_fft, (complex_result, size, fsize)


def pairwise_correlate(X, Y, mode='full', fast=True,
                       fft_precomputed=False, fft_info=None):
    """
    Parameters
    ----------
    X, Y: array_like
        Two-dimensional arrays to convolve
    mode: string
        ["full"|"valid"|"same"]

    Other Parameters
    ----------------
    fast : bool
        if True (default) use a fast broadcasting-based algorithm.
        This is mainly for unit-testing purposes.
    fft_precomputed : bool
        if True, then X and Y actually contain the pre-computed FFT
        of X and Y. Default is False. Cannot be used with fast=False.
        FFTs can be precomputed with the precompute_fft() function.
        If True, then complex_result must be specified.
    fft_info : bool
        Required if fft_precomputed==True.

    Returns
    -------
    out: array
        Three-dimensional array. out[i, j] contains the
        convolution of X[i] and Y[j]
    """
    if not fast:
        if fft_precomputed:
            raise ValueError("Cannot have fft_precomputed and not fast")
        return pairwise_correlate_slow(X, Y, mode)

    if mode != 'full':
        raise NotImplementedError()

    if fft_precomputed:
        if fft_info is None:
            raise ValueError("must specify complex_result=[True/False] "
                             "if fft_precomputed is True")
        X_fft, Y_fft = np.asarray(X), np.asarray(Y)
    else:
        X_fft, Y_fft, fft_info = precompute_fft(X, Y)

    complex_result, size, fsize = fft_info

    assert X_fft.ndim == 2
    assert Y_fft.ndim == 2
    assert X_fft.shape[-1] == Y_fft.shape[-1]

    # prepare broadcasting
    X_fft = X_fft[:, np.newaxis, :]
    Y_fft = Y_fft[np.newaxis, :, :]

    if not complex_result:
        M = np.fft.irfftn(X_fft * Y_fft, fsize)[:, :, :size].real
    else:
        M = np.fft.ifftn(X_fft * Y_fft, fsize)[:, :, :size]

    #if mode == "full":
        #pass
    #elif mode == "same":
        #return _centered(ret, s1)
    #elif mode == "valid":
        #return _centered(ret, s1 - s2 + 1)

    return M


def _batch_crosscorr(X, Y, batch_size, reduce_func,
                     fft_precomputed=False, fft_info=None):
    """Helper routine for batch fft-based cross-correlation.

    Parameters
    ----------
    X : array_like
        shape = [Nx, n_features]
    Y : array_like
        shape = [Ny, n_features]
    batch_size : integer
        perform computation in batches of this size.
    reduce_func: function
        a function which will reduce the input along its last axis.
        Input is the result of pairwise_correlate() on the batch, and is
        of shape (n, m, p). reduce_func should take this as input and return
        a suitable array of shape (n, m).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    assert X.ndim == 2
    assert Y.ndim == 2

    # precompute fft if necessary
    if fft_precomputed:
        Xfft, Yfft = X, Y
        assert fft_info is not None
    else:
        Xfft, Yfft, fft_info = precompute_fft(X, Y)

    # if batches are unnecessary, do the calculation in one step
    if batch_size is None or X.shape[0] * Y.shape[0] <= batch_size:
        M = pairwise_correlate(Xfft, Yfft, fft_info=fft_info,
                               fft_precomputed=True)
        return reduce_func(M)

    # otherwise, we divide the computation into batches
    result = np.zeros((X.shape[0], Y.shape[0]))

    if Y.shape[0] < batch_size:
        batchsize = [batch_size // Y.shape[0], Y.shape[0]]
    elif X.shape[0] < batch_size:
        batchsize = [X.shape[0], batch_size // X.shape[0]]
    else:
        batchsize = 2 * [int(np.sqrt(batch_size))]

    nbatches = [1 + (X.shape[0] - 1) // batchsize[0],
                1 + (Y.shape[0] - 1) // batchsize[1]]

    for i in range(nbatches[0]):
        sliceX = slice(i * batchsize[0], (i + 1) * batchsize[0])
        for j in range(nbatches[1]):
            sliceY = slice(j * batchsize[1], (j + 1) * batchsize[1])
            corr = pairwise_correlate(Xfft[sliceX], Yfft[sliceY],
                                      fft_precomputed=True, fft_info=fft_info)
            result[sliceX, sliceY] = reduce_func(corr)
    return result


def crosscorr_similarity(X, Y, batch_size=10000):
    """Cross-correlation similarity between X and Y

    Parameters
    ----------
    X : array_like
        shape = [Nx, n_features]
    Y : array_like
        shape = [Ny, n_features]
    batch_size : integer (default=10000)
        perform computation in batches of this size.

    Returns
    -------
    M : np.ndarray
        the pairwise cross-correlation kernel between X and Y, shape [Nx, Ny]
    """
    reduce_func = lambda corr: corr.max(-1)
    return _batch_crosscorr(X, Y, batch_size, reduce_func)


def crosscorr_kernel(X, Y, lambda_=10, batch_size=10000):
    """Cross-correlation kernel between X and Y

    Parameters
    ----------
    X : array_like
        shape = [Nx, n_features]
    Y : array_like
        shape = [Ny, n_features]
    lambda_ : float (default=10)
        the exponential free parameter in the kernel.
    batch_size : integer (default=10000)
        perform computation in batches of this size.

    Returns
    -------
    M : np.ndarray
        the pairwise cross-correlation kernel between X and Y, shape [Nx, Ny]
    """
    reduce_func = lambda corr, lambda_=lambda_: np.exp(lambda_ * corr).sum(-1)
    return _batch_crosscorr(X, Y, batch_size, reduce_func)


def crosscorr_metric(X, Y, batch_size=10000):
    # This could be done WAY more efficiently, especially the XX & YY terms
    XX = crosscorr_similarity(X, X, batch_size)
    YY = crosscorr_similarity(Y, Y, batch_size)
    XY = crosscorr_similarity(X, Y, batch_size)
    return XX.diagonal()[:, None] + YY.diagonal() - 2 * XY
