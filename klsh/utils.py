import itertools
import numpy as np


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

    # make sure we have a C-ordered buffer
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
