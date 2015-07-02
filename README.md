# kmeans

A simple and fast K-means implementation for Python.

The speed comes from using a Cython .pyx module to compute
distances (in parallel).  I use pyximport to do a build-and
-import, so Cython needs to be installed for the package to
work.  On the plus side, there is no need to manually build
anything.  Just import kmeans and go.

### EXAMPLE

```
import kmeans

data = np.random.rand(500, 2)
centers = kmeans.kmeans(data, 10)
labels = kmeans.assignments(data, centers)
clusters = kmeans.cluster(data, centers)
```

