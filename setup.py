from distutils.core import setup
from Cython.Build import cythonize

import os

import numpy as np

# save current working directory; switch into the directory containing the setup.py file
odir = os.getcwd()
os.chdir( os.path.dirname( os.path.realpath(__file__) ) )

module_list = ['*.pyx']

include_dirs = [
    '.',                    # not necessary at the moment
    #np.get_include(),       # does not seem to be necessary...
    #'/usr/local/lib/python2.7/dist-packages/Cython/Includes/libcpp/'    # to get rid of an error/warning message (might be a bug in Cython)
    ]

ext_modules = cythonize(module_list, nthreads=8, include_path=include_dirs, build_dir='build/')
for m in ext_modules:
    # Remove the "kmeans." from each extension module name.
    # This ensures that the .so files get put in the right place.
    # Typically, the setup.py file would reside one directory below
    # the kmeans/ package. Things get messed up because we want to
    # place setup.py inside the package.  This hack fixes things.
    print m.name
    m.name = m.name[7:]

setup( 
    ext_modules=ext_modules,
    )

# restore the current working directory
os.chdir(odir)


