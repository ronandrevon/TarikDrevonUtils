import os
import numpy as np
from subprocess import check_output
from pytest_html import extras
from os.path import exists
from . import displayStandards as dsp
# import pytest


def get_path(file):
    dir=os.path.dirname(file)
    filename=os.path.basename(file).replace('.py','')
    ref=os.path.join(dir,'ref',filename)
    out=os.path.join(dir,'out',filename)

    if not exists(ref):print(check_output('mkdir -p %s' %ref,shell=True).decode())
    if not exists(out):print(check_output('mkdir -p %s' %out,shell=True).decode())

    return out,ref,dir

def add_link(file):
    out,ref,dir=get_path(file)
    def link_dec(test_f):
        def test_g(extra):
            fig,ax=test_f()

            name=os.path.join(out,test_f.__name__+'.png')
            ax.name=name
            # dir = 'file://'
            dsp.disp_quick(name,ax,opt='sc')

            # link='file://%s' %name
            #assuming 8010 serves tests
            hostname=check_output('hostname -A', shell=1).decode().strip().split()[-1]
            figpath=name.split('/tests/')[-1]
            link='http://%s:8010/%s' %(hostname,figpath);print(link)
            extra.append(extras.image(link ))
            extra.append(extras.url(  link ))

        return test_g
    return link_dec

def cmp_ref(file,tol=1e-8):
    out,ref,dir=get_path(file)
    def cmp_dec(test_f):
        name=test_f.__name__+'.npy'
        def test_g(*args):
            a_out = test_f(*args)

            a_out_str = os.path.join(out,name)
            a_ref_str = os.path.join(ref,name)
            np.save(a_out_str,a_out)
            if not exists(a_ref_str):
                np.save(a_ref_str,a_out)
                print('No data in ref. First time test is done')

            a_ref = np.load(a_ref_str)
            assert abs(a_out-a_ref).sum()<tol
        return test_g
    return cmp_dec
