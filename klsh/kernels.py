import numpy as np
from scipy import fftpack, signal


def pairwise_correlate(X, Y, mode='full', fast=True):
    """
    Parameters
    ----------
    X, Y: array_like
        Two-dimensional arrays to convolve
    mode: string
        ["full"|"valid"|"same"]
        
    Returns
    -------
    out: array
        Three-dimensional array. out[i, j] contains the
        convolution of X[i] and Y[j]
    """
    X, Y = map(np.atleast_2d, (X, Y))
    
    Y = Y[:, ::-1]
    
    if mode != 'full':
        raise NotImplementedError()
    
    assert X.ndim == 2
    assert Y.ndim == 2
    
    if fast:
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
            M = np.fft.irfftn(X_fft[:, np.newaxis, :] * Y_fft[np.newaxis, :, :],
                              fsize)[:, :, :size].real
        else:
            X_fft = np.fft.fftn(X, fsize)
            Y_fft = np.fft.fftn(Y, fsize)
            M = np.fft.ifftn(X_fft[:, np.newaxis, :] * Y_fft[np.newaxis, :, :],
                             fsize)[:, :, :size]

        #if mode == "full":
            #pass
        #elif mode == "same":
            #return _centered(ret, s1)
        #elif mode == "valid":
            #return _centered(ret, s1 - s2 + 1)
        
    else:
        # Kept for testing purposes
        first_result = signal.fftconvolve(X[0], Y[0], mode)
        M = np.zeros((X.shape[0], Y.shape[0], len(first_result)),
                     dtype=first_result.dtype)
    
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                M[i, j] = signal.fftconvolve(X[i], Y[j], mode)
    return M


def crosscorr_kernel(X, Y, lambda_=100):
    M = pairwise_correlate(X, Y)
    return np.sum(M ** lambda_, -1)
