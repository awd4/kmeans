import numpy as np
cimport numpy as cnp

cimport cython
from cython cimport parallel


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _sq_dist(double[::1] x, double[::1] y) nogil:
    cdef:
        size_t i, n
        double sd, diff
    n = x.shape[0]
    if n != y.shape[0] or n <= 0:
        return -1.0
    sd = 0.0
    for i in range(n):
        diff = x[i] - y[i]
        sd += diff * diff
    return sd


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _sq_dist_ptr(double *x, double *y, size_t n) nogil:
    cdef:
        size_t i
        double sd, diff
    sd = 0.0
    for i in range(n):
        diff = x[i] - y[i]
        sd += diff * diff
    return sd


@cython.boundscheck(False)
@cython.wraparound(False)
def _sq_distances(double[:,::1] data, double[:,::1] cntrs, double[:,::1] dists=None):
    cdef:
        int i
        size_t j, k, n
    n = data.shape[0]
    k = cntrs.shape[0]
    if dists is None:
        dists = np.zeros((n, k))
    for i in parallel.prange(n, nogil=True, num_threads=8):
        for j in range(k):
            dists[i,j] = _sq_dist(data[i,:], cntrs[j,:])
            #dists[i,j] = _sq_dist_ptr(&data[i,0], &cntrs[j,0], k)
    return np.array(dists)




def test_sq_distances():
    data = np.array( [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]] )
    cntrs = np.array([[0.5, 0.0], [1.0, 2.0]])
    dist = np.array( [
            [0.25, 5.0],
            [0.25, 4.0],
            [1.25, 1.0],
            [6.25, 1.0]
            ] )
    sd = _sq_distances(data, cntrs)
    assert np.allclose( sd, dist ) 


