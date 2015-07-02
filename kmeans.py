import pyximport; pyximport.install()
import cykmeans

import numpy as np


def kmeans(data, k, iters=20, reps=1):
    # TODO: prefer candidate centers that have exactly k centers
    data = np.array(data, dtype=float)
    cntrs = _kmeans(data, k, iters)
    best = _score(data, cntrs)
    for r in range(1, reps):
        c = _kmeans(data, k, iters)         # candidate centers
        s = score(data, c)
        if s < best:
            best = s
            cntrs = c
    return cntrs


def assignments(data, cntrs, dists=None):
    d = cykmeans._sq_distances(data, cntrs, dists)
    a = d.argmin(axis=1)
    return a


def cluster(data, cntrs):
    a = assignments(data, cntrs)
    c = []
    for i in range(len(cntrs)):
        ai = a == i
        if True not in ai:
            continue
        c.append( data[ai] )
    return c


def _score(data, cntrs):
    d = cykmeans._sq_distances(data, cntrs)
    return d.min(axis=1).sum()


def _kmeans(data, k, iters=20):
    assert data.dtype == np.float64 and k >= 1 and iters >= 0
    n = len(data)
    cntrs = data[np.random.randint(n, size=k)] 
    cntrs += np.random.randn(*cntrs.shape) / 1000.
    dists = np.zeros((n, k))
    prev_a = None
    valid = np.ones(k)
    for i in range(iters):
        valid[:] = 1
        a = assignments(data, cntrs, dists)
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
    centers = kmeans(data, 10)
    labels = assignments(data, centers)
    clusters = cluster(data, centers)


