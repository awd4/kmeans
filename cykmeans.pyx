import numpy as np
from libc.math cimport sqrt as csqrt

cimport cython
from cython cimport parallel


@cython.boundscheck(False)
@cython.wraparound(False)
def _sq_dist(double[::1] x, double[::1] y):
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


def _sq_distances(double[:,::1] data, double[:,::1] cntrs, double[:,::1] dists=None):
    if dists is None:
        dists = np.zeros((data.shape[0], cntrs.shape[0]))
    _c_sq_distances(data, cntrs, dists)
    return np.array(dists)


def _distances(double[:,::1] data, double[:,::1] cntrs, double[:,::1] dists=None):
    if dists is None:
        dists = np.zeros((data.shape[0], cntrs.shape[0]))
    _c_distances(data, cntrs, dists)
    return np.array(dists)


@cython.boundscheck(False)
@cython.wraparound(False)
def _adjust_centers(double[:,::1] data, double[:,::1] cntrs, long[::1] a, long[::1] valid):
    cdef:
        size_t i, j, k, l, m, n
        long q
    n = data.shape[0]
    m = data.shape[1]
    k = cntrs.shape[0]
    for i in range(k):
        valid[i] = 0
        for j in range(m):
            cntrs[i,j] = 0.0
    for i in range(n):
        q = a[i]
        valid[q] += 1
        for j in range(m):
            cntrs[q,j] += data[i,j]
    for i in range(k):
        if valid[i] > 0:
            for j in range(m):
                cntrs[i,j] /= valid[i]
        else:
            l = np.random.randint(data.shape[0])
            for j in range(m):
                cntrs[i,j] = data[l,j]


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
cdef void _c_sq_distances(double[:,::1] data, double[:,::1] cntrs, double[:,::1] dists):
    cdef:
        int i
        size_t j, k, n, m
    n = data.shape[0]
    m = data.shape[1]
    k = cntrs.shape[0]
    for i in parallel.prange(n, nogil=True, num_threads=8):
    #for i in range(n):
        for j in range(k):
            dists[i,j] = _sq_dist_ptr(&data[i,0], &cntrs[j,0], m)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_distances(double[:,::1] data, double[:,::1] cntrs, double[:,::1] dists):
    cdef:
        size_t i, j
    _c_sq_distances(data, cntrs, dists)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            dists[i, j] = csqrt(dists[i,j])


@cython.boundscheck(False)
@cython.wraparound(False)
def _elkan(double[:,::1] cntrs, double[:,::1] data, double[:,::1] lb, double[::1] ub, double[::1] cu, double[:,::1] cc, long[::1] ca, double[:,::1] pc):
    cdef:
        int i
        size_t j, n, m
        size_t k
        long q
        bint upper_tight
    n = data.shape[0]
    m = data.shape[1]
    k = cc.shape[0]
    # Update lower and upper bounds
    for i in range(k):
        cu[i] = csqrt( _sq_dist_ptr(&pc[i,0], &cntrs[i,0], m) )
    for i in range(n):
        for j in range(k):
            lb[i, j] -= cu[j]
    for i in range(n):
        q = ca[i]
        ub[i] += cu[q]
    # Re-assign points to centers (if needed)
    _c_distances(cntrs, cntrs, cc)
    for i in parallel.prange(n, nogil=True, num_threads=8):
    #for i in range(n):
        q = ca[i]
        for j in range(k):
            if q == j:
                continue
            if ub[i] > 0.5 * cc[q, j]:
                break
        else:   # no break happened in the for-loop
            continue    # skip this point; its center does not change
        upper_tight = False
        for j in range(k):
            if q == j:
                continue
            if ub[i] <= 0.5 * cc[q, j] or ub[i] <= lb[i, j]:
                continue
            if not upper_tight:
                lb[i, q] = csqrt( _sq_dist_ptr( &data[i,0], &cntrs[q,0], m ) )
                ub[i] = lb[i, q]
                upper_tight = True
                if ub[i] <= 0.5 * cc[q, j] or ub[i] <= lb[i, j]:
                    continue
            lb[i, j] = csqrt( _sq_dist_ptr( &data[i,0], &cntrs[j,0], m ) )
            if lb[i, j] <= ub[i]:
                ca[i] = j
                q = j
                ub[i] = lb[i, j]




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
    cntrs = np.random.rand(100, 2)
    sd = _sq_distances(data, cntrs)
    assert not np.any( np.isnan(sd) )

def test_adjust_centers():
    data = np.array( [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 0.0]] )
    cntrs = np.zeros( (3,2) )
    a = np.array( [0, 2, 2, 2], dtype=int )
    valid = np.zeros( cntrs.shape[0], dtype=int )
    _adjust_centers(data, cntrs, a, valid)
    assert not np.any( np.isnan( np.array(cntrs) ) )


