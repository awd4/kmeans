# kmeans

A fast K-means implementation for Python.

The speed comes from using a Cython .pyx module to compute
distances (in parallel).  Cython needs to be installed for
the package to work, since I use the cythonize() function
to compile the .pyx module.  I have written the `__init__.py`
module to automatically call the setup.py module, which takes
care of building the cykmeans.so file.  So there is no need
to manually build anything.  Just import kmeans and go.

### INSTALL

Currently there are two recommended ways of using kmeans.
The first is to copy the code to a convenient location
(e.g., into the root directory of your code).

The second is to do a user install, which places the kmeans
package in the user-site directory.  To do that, run:
```
python setup.py user_install
```
This is a non-standard way to distribute a package (it's not
up on PyPI and not pip-installable).  Perhaps I will change
that later, but for now it is what it is.

### EXAMPLE

```
import kmeans

data = np.random.rand(500, 2)
cntrs = kmeans.centers(data, 10)
labels = kmeans.assign(data, cntrs)
clusters = kmeans.cluster(data, cntrs)
```

