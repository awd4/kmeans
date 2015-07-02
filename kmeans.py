import pyximport; pyximport.install()
import cykmeans

import numpy as np


def centers(data, k, iters=20, reps=1, alg='lloyd'):
    # TODO: prefer candidate centers that have exactly k centers
    data = np.require(data, np.float64, 'C')
    best = float('inf')
    best_cntrs = None
    for r in range(reps):
        cntrs = _init(data, k)
        assigner = _assigner_factory(alg, data, cntrs)
        c = _kmeans(data, k, cntrs, assigner, iters)    # candidate centers
        s = _score(data, c)
        if s < best:
            best = s
            best_cntrs = c
    return best_cntrs


def assign(data, cntrs):
    data = np.require(data, np.float64, 'C')
    d = cykmeans._sq_distances(data, cntrs)
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
    n = len(data)
    cntrs = data[np.random.randint(n, size=k)] 
    return cntrs


def _assigner_factory(alg, data, cntrs):
    sub_factory = {
            'lloyd':_lloyd_assigner,
            'elkan':_elkan_assigner,
            }[alg]
    return sub_factory(data, cntrs)


def _lloyd_assigner(data, cntrs):
    data = np.require(data, np.float64, 'C')
    dists = np.zeros((len(data), len(cntrs)))
    def lloyd(cntrs):
        d = cykmeans._sq_distances(data, cntrs, dists)
        a = d.argmin(axis=1)
        return a
    return lloyd


class _elkan_assigner(object):
    def __init__(self, data, cntrs):
        self.data = np.require(data, np.float64, 'C')
        self.k = len(cntrs)
        self.lb = np.sqrt( cykmeans._sq_distances(data, cntrs) )    # lower-bounds
        self.ub = self.lb.min(axis=1)                               # upper-bounds
        self.cc = np.zeros((self.k, self.k))                        # center-to-center distances
        self.ca = self.lb.argmin(axis=1)                            # current assignments
        self.pc = cntrs.copy()                                      # centers from previous iteration

    def __call__(self, cntrs):
        # Update lower and upper bounds
        data, k, lb, ub, cc, ca, pc = self.data, self.k, self.lb, self.ub, self.cc, self.ca, self.pc
        cdist = np.sqrt( np.array( [cykmeans._sq_dist( self.pc[i,:], cntrs[i,:] ) for i in range(self.k)] ) )
        lb -= cdist[None, :]
        for i, x in enumerate(data):
            q = ca[i]
            ub[i] += cdist[q]
        # Re-assign points to centers (if needed)
        cc[:] = np.sqrt( cykmeans._sq_distances(cntrs, cntrs) )
        for i, x in enumerate(data):
            q = ca[i]
            for j in range(self.k):
                if q == j:
                    continue
                if ub[i] > 0.5 * cc[q, j]:
                    break
            else:   # no break happened in the for-loop
                continue    # skip this point; its center does not change
            upper_tight = False
            for j, c in enumerate(cntrs):
                if q == j:
                    continue
                if ub[i] <= 0.5 * cc[q, j] or ub[i] <= lb[i, j]:
                    continue
                if not upper_tight:
                    lb[i, q] = np.sqrt( cykmeans._sq_dist( x, cntrs[q,:] ) )
                    ub[i] = lb[i, q]
                    upper_tight = True
                    if ub[i] <= 0.5 * cc[q, j] or ub[i] <= lb[i, j]:
                        continue
                lb[i, j] = np.sqrt( cykmeans._sq_dist( x, cntrs[j,:] ) )
                if lb[i, j] <= ub[i]:
                    ca[i] = j
                    q = j
                    ub[i] = lb[i, j]
        pc = cntrs.copy()
        return ca


def _score(data, cntrs):
    data = np.require(data, np.float64, 'C')
    d = cykmeans._sq_distances(data, cntrs)
    return d.min(axis=1).sum()


def _kmeans(data, k, cntrs, assigner, iters=20):
    assert data.dtype == np.float64 and k >= 1 and iters >= 0
    n = float(len(data))
    prev_a = None
    valid = np.ones(k)
    for i in range(iters):
        valid[:] = 1
        a = assigner(cntrs)
        for j in range(k):
            aj = a==j
            if True in aj:
                cntrs[j,:] = data[a==j,:].mean()
            else:
                valid[j] = 0
        # stop updating the centers if less than 1% of the points have changed clusters
        changed = np.sum( prev_a != a ) if prev_a is not None else n
        if changed / n < 0.01:
            break
        prev_a = a.copy()
    cntrs = cntrs[valid==1]
    return cntrs



def test_cykmeans():
    cykmeans.test_sq_distances()
    cykmeans.test_default()

def test_kmeans():
    data = np.random.rand(500, 2)
    cntrs = centers(data, 10)
    labels = assign(data, cntrs)
    clusters = cluster(data, cntrs)

def test_elkan():
    n = 500
    k = 10
    np.random.seed(12345)
    d1 = np.random.rand(n, 2)
    c1 = centers(d1, k, alg='lloyd')
    l1 = assign(d1, c1)
    u1 = cluster(d1, c1)
    return
    print k
    print d1.shape
    print c1.shape
    np.random.seed(12345)
    d2 = np.random.rand(n, 2)
    c2 = centers(d2, k, alg='elkan')
    print k
    print d2.shape
    print c2.shape
    l2 = assign(d2, c2)
    return
    u2 = cluster(d2, c2)
    assert np.all( d1 == d2 )
    assert np.all( c1 == c2 )
    assert np.all( l1 == l2 )
    assert all( [np.all( e1 == e2 ) for e1, e2 in zip(u1, u2)] )


