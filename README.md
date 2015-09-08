# kmeans

A fast K-means implementation for Python.

The speed comes from using a Cython .pyx module to compute
distances (in parallel).  Cython needs to be installed for
the package to work, since I use the cythonize() function
to compile the .pyx module.  I have written the `__init__.py`
module to automatically call the setup.py module, which takes
care of building the cykmeans.so file.  So there is no need
to manually build anything.  Just import kmeans and go.

### EXAMPLE

```
import kmeans

data = np.random.rand(500, 2)
cntrs = kmeans.centers(data, 10)
labels = kmeans.assign(data, cntrs)
clusters = kmeans.cluster(data, cntrs)
```

