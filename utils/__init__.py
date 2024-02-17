#print('importing...')
import os
from . import _version

def get_version(changelog):
    '''automatically retrieves the version from changelog.md
    Parameters :
    ------------
    folder where the changelog.md is'''
    with open(changelog,'r') as f:
        version=''
        while not version:
            line = f.readline()
            if line.startswith('##') : #and 'dev' not in line:
                version = line.replace('##','').strip()
    return version

def _set_version(changelog,version):
    '''automatically retrieves the version from changelog.md
    and  writes it into  _version.py
    '''
    s = """version='{version}'
    """.format(version=version)

    version_file=os.path.join(os.path.dirname(__file__),'_version.py')
    with open(version_file,'w') as f:
        f.write(s)
    print('_version.py updated with version %s' %version)

def _get_version(dir):
    ''' get the current version from either _version or changelog'''
    changelog = os.path.join(dir,"changelog.md")
    if os.path.exists(changelog):
        version=get_version(changelog)
        if not _version.version == version:
            _set_version(changelog,version)
        return version
    else:
        return _version.version

__version__=_get_version(os.path.join(os.path.dirname(__file__),'..'))

# py standard library
#from math import*
import importlib as imp
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spe
import pandas as pd


# specific libraries
from . import displayStandards as dsp
from . import regionManager as regMg
from . import materialManager as matMg
from . import newtonRaphson as nwt
from . import bissection as bis
from . import commonInputOutput as iostd
from . import physicsConstants as cst
from . import glob_colors  as colors
from .glob_colors import *

def funcs(obj):
    import inspect
    functions = '\n'.join([f[0] for f in inspect.getmembers(obj,predicate=inspect.isfunction)])
    print(functions)
