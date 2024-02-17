import numpy as np
from . import physicsConstants as cst

def Dmin(L,m,smax=400):
    '''
    smax:strength in MPa
    '''
    smax_x = {
        'wood'       : 35,
        'mild_steel' : 400,
    }
    if isinstance(smax,str):
        smax=smax_x[smax]
    return np.sqrt(L*m*cst.g0*4/(np.pi*smax*1e6))*1e3*2
