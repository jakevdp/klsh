import numpy as np
from scipy import fftpack, signal


def pairwise_correlate_slow(X, Y, mode='full'):
    X, Y = map(np.atleast_2d, (X, Y))
    assert X.ndim == 2
    assert Y.ndim == 2
    
    Y = Y[:, ::-1]

    # Kept for testing purposes
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


def crosscorr_kernel(X, Y, lambda_=100):
    M = pairwise_correlate(X, Y)
    return np.sum(M ** lambda_, -1)
