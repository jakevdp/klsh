from numpy.testing import assert_allclose
from ..utils import hamming_hashes

def test_hamming_hashes():
    assert_allclose(list(hamming_hashes(0, 3)),
                    [0, 1, 2, 4, 3, 5, 6, 7])
    assert_allclose(list(hamming_hashes(1, 3)),
                    [1, 0, 3, 5, 2, 4, 7, 6])
