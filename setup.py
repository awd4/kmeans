from distutils.core import setup
from Cython.Build import cythonize

import os


def build():
    # save current working directory; switch into the directory containing the setup.py file
    odir = os.getcwd()
    os.chdir( os.path.dirname( os.path.realpath(__file__) ) )

    module_list = ['*.pyx']

    ext_modules = cythonize(module_list, nthreads=8, build_dir='build/')
    for m in ext_modules:
        # Remove the "kmeans." from each extension module name.
        # This ensures that the .so files get put in the right place.
        # Typically, the setup.py file would reside one directory below
        # the kmeans/ package. Things get messed up because we want to
        # place setup.py inside the package.  This hack fixes things.
        m.name = m.name[7:]

    setup( 
        ext_modules=ext_modules,
        )

    # restore the current working directory
    os.chdir(odir)

def user_install():
    import site
    import shutil
    sdir = site.getusersitepackages()                       # site directory
    idir = os.path.join( sdir, 'kmeans' )                   # install directory
    cdir = os.path.dirname( os.path.realpath(__file__) )    # code directory
    if os.path.exists( idir ):
        reply = raw_input('Overwrite ' + idir + ' (y/n)?')
        if reply in ['y', 'Y', 'yes', 'Yes']:
            shutil.rmtree( idir )
        else:
            return
    os.mkdir( idir )
    for n in ['cykmeans.pyx', '__init__.py', 'kmeans.py', 'LICENSE', 'README.md', 'setup.py', 'test.py']:
        path = os.path.join( cdir, n )
        shutil.copy( path, idir )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1].startswith('build'):
            build()
        elif sys.argv[1] == 'user_install':
            user_install()


