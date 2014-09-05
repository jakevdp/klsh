import itertools
import contextlib
import time
import numbers
import numpy as np


@contextlib.contextmanager
def timeit(fmt=None):
    if fmt is None:
        fmt = "{0:.2g} sec"
    t0 = time.time()
    yield
    t1 = time.time()
    print(fmt.format(t1 - t0))


def create_rng(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    Adapted from sklearn.utils.check_random_state()
    """
    if seed is None or seed is np.random:
        rng = np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer)):
        rng = np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        raise ValueError('{0} cannot be used to seed a '
                         'numpy.random.RandomState instance'.format(seed))
    return rng


def packbits_axis(X, axis=-1):
    """Create a compact representation of rows of bits in numpy

    Parameters
    ----------
    X : array_like
        a d-dimensional array whose rows will be treated as a sequence of bits
    axis : integer
        the axis along which to pack the bits (default=-1)

    Returns
    -------
    x : array_like
        a (d - 1)-dimensional structured array containing sets of 8-bit
        integers which compactly represent the bits along the specified
        axis of X.
    """
    X = np.asarray(X, dtype=np.uint8)

    # roll specified axis to the back
    if axis not in (-1, X.ndim - 1):
        X = np.rollaxis(X, axis).transpose(list(range(1, X.ndim)) + [0])

    # make sure we have a C-ordered contiguous buffer
    X = np.asarray(X, order='C')
    bits = np.packbits(X, -1)

    return_shape = bits.shape[:-1]
    return_type = [('', 'u1') for i in range(bits.shape[-1])]

    return np.ndarray(return_shape, dtype=return_type, buffer=bits)


def unpackbits_axis(x, axis=-1, axissize=None):
    """Inverse of packbits_axis

    Parameters
    ----------
    x : ndarray
        record array of any shape, with multiple data of type uint8
    axissize : integer
        max size of expanded axis. Default is 8 * len(x.dtype)

    Returns
    -------
    X : ndarray
        array of shape x.shape[:axis] + (8 * d,) + x.shape[axis:]
        where d is the number of unsigned ints in each element of the
        record array.
    """
    assert all(x.dtype[i] == np.uint8 for i in range(len(x.dtype)))
    X = np.ndarray(x.shape + (len(x.dtype),),
                   dtype=np.uint8,
                   buffer=x)
    X = np.unpackbits(X, -1)

    if axissize is not None:
        slices = [slice(None) for i in range(X.ndim)]
        slices[-1] = slice(0, axissize)
        X = X[slices]

    return np.rollaxis(X, -1, axis)


def hamming_cdist(x, y=None, use_broadcasting=False):
    """Compute the matrix of hamming distances between x and y, which are
    stored in packed-bit format.

    Parameters
    ----------
    x, y: nd_arrays
        x and y should be one-dimensional structured arrays with data type
        made of some number of unsigned integers.
    """
    # TODO: make work with types other than uint8? maybe not needed.
    x = np.atleast_1d(x)
    assert x.ndim == 1

    if len(x.dtype) > 0:
        nbytes = len(x.dtype)
        assert all(x.dtype[i] == np.uint8 for i in range(nbytes))
    else:
        nbytes = 1
        assert x.dtype == np.uint8

    if y is None:
        y = x
    else:
        y = np.atleast_1d(y)
        assert y.ndim == 1
        assert y.dtype == x.dtype

    if use_broadcasting:
        x_ints = np.ndarray((x.shape[0], nbytes),
                            dtype=np.uint8,
                            buffer=x.data)
        if y is x:
            y_ints = x_ints
        else:
            y_ints = np.ndarray((y.shape[0], nbytes),
                                dtype=np.uint8,
                                buffer=y.data)

        nonmatch_matrix = np.bitwise_xor(x_ints[:, np.newaxis, :],
                                         y_ints[np.newaxis, :, :])

        res = np.unpackbits(nonmatch_matrix[:, :, :, None], -1).sum((2, 3))
    else:
        if len(x.dtype) > 0:
            it = (np.unpackbits(np.bitwise_xor(x[d][:, None],
                                               y[d])[:, :, None], -1).sum(-1)
                  for d in x.dtype.names)
            res = sum(it)
        else:
            res = np.unpackbits(np.bitwise_xor(x[:, None],
                                               y)[:, :, None], -1).sum(-1)

    return res


def hamming_hashes(hashval, nbits, nmax=None):
    """Return an iterator over all (integer) hashes,
    in order of hamming distance

    Parameters
    ----------
    hashval : integer
        hash value to match
    nbits : integer
        number of bits in the hash
    nmax : integer (optional)
        if specified, halt the iterator after given number of results
    """
    if nmax is not None:
        return itertools.islice(hamming_hashes(hashval, nbits), nmax)
    else:
        hashval = int(hashval)
        bits = [2 ** i for i in range(nbits)]
        return (hashval ^ sum(flip)
                for nflips in range(nbits + 1)
                for flip in itertools.combinations(bits, nflips))
