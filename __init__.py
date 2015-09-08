# Automatically compile the cykmeans module if it hasn't been compiled yet.
import subprocess
import os
dirname = os.path.dirname( os.path.realpath(__file__) )
setup_path = os.path.join( dirname, 'setup.py' )
subprocess.check_call(['python', setup_path, 'build_ext', '--inplace'])
from kmeans import *
