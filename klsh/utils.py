import itertools


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
