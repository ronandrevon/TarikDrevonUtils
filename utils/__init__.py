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
    '''writes version into into  <dir>/_version.py
    where dir is the location of changelog
    '''
    s = """version='{version}'
    """.format(version=version)
    dir = os.path.dirname(changelog)
    version_file=os.path.join(dir,'_version.py')
    with open(version_file,'w') as f:
        f.write(s)

def _check_version(dir,version):
    ''' check the current version from <version> is identical to the one in changelog'''
    changelog = os.path.join(dir,"changelog.md")
    if os.path.exists(changelog):
        new_version=get_version(changelog)
        # print(_version.version,version,_version.version == version)
        if not (new_version == version):
            print('old version "%s". Updating _version.py with version "%s".' %(version,new_version))
            _set_version(changelog,version)
        return version

__version__=_check_version(os.path.join(os.path.dirname(__file__)),_version.version)

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
