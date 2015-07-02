import pyximport; pyximport.install()
import cykmeans

import numpy as np


def centers(data, k, iters=20, reps=1, alg='lloyd'):
    # TODO: prefer candidate centers that have exactly k centers
    data = np.require(data, np.float64, 'C')
    cntrs = _kmeans(data, k, iters, alg)
    best = _score(data, cntrs)
    for r in range(1, reps):
        c = _kmeans(data, k, iters, alg)    # candidate centers
        s = score(data, c)
        if s < best:
            best = s
            cntrs = c
    return cntrs


def assign(data, cntrs, dists=None):
    data = np.require(data, np.float64, 'C')
    d = cykmeans._sq_distances(data, cntrs, dists)
    a = d.argmin(axis=1)
    return a


def cluster(data, cntrs):
    a = assign(data, cntrs)
    c = []
    for i in range(len(cntrs)):
        ai = a == i
        if True not in ai:
            continue
        c.append( data[ai] )
    return c


def _init(data, k):
    cntrs = data[np.random.randint(n, size=k)] 
    cntrs += np.random.randn(*cntrs.shape) / 1000.
    return cntrs


def _score(data, cntrs):
    d = cykmeans._sq_distances(data, cntrs)
    return d.min(axis=1).sum()


def _kmeans(data, k, iters=20, alg='lloyd'):
    assert data.dtype == np.float64 and k >= 1 and iters >= 0
    n = len(data)
    cntrs = _init(data, k)
    dists = np.zeros((n, k))
    prev_a = None
    valid = np.ones(k)
    for i in range(iters):
        valid[:] = 1
        a = assign(data, cntrs, dists)
        for j in range(k):
            aj = a==j
            if True in aj:
                cntrs[j,:] = data[a==j,:].mean()
            else:
                valid[j] = 0
        # stop updating the centers if less than 1% of the points have changed clusters
        changed = np.sum( prev_a != a ) if prev_a is not None else len(data)
        if changed / float(len(data)) < 0.01:
            break
        prev_a = a.copy()
    cntrs = cntrs[valid==1]
    return cntrs



def test_cykmeans():
    cykmeans.test_sq_distances()

def test_kmeans():
    data = np.random.rand(500, 2)
    cntrs = centers(data, 10)
    labels = assign(data, cntrs)
    clusters = cluster(data, cntrs)

def test_elkan():
    np.random.seed(12345)
    d1 = np.random.rand(500, 2)
    c1 = centers(d1, 10, alg='elkan')
    l1 = assign(d1, c1)
    u1 = cluster(d1, c1)
    np.random.seed(12345)
    d2 = np.random.rand(500, 2)
    c2 = centers(d2, 10, alg='elkan')
    l2 = assign(d2, c2)
    u2 = cluster(d2, c2)
    assert np.all( d1 == d2 )
    assert np.all( c1 == c2 )
    assert np.all( l1 == l2 )
    assert all( [np.all( e1 == e2 ) for e1, e2 in zip(u1, u2)] )


