import importlib as imp
import numpy as np
from numpy import pi
import scipy.optimize as sci_opt
import utils.newtonRaphson as nwt
import utils.displayStandards as dsp
imp.reload(nwt)
imp.reload(dsp)

def test_newton_rec():
    f = lambda y,x : np.tan(pi/2*(y-x))
    df = lambda y,x : pi/2*(1+f(y,x)**2)
    cond = lambda x0,x1,y0,y1 : y1>y0

    x,y0 = np.array([0,1.6]),0.01
    y0s = nwt.newton_recursive(f,df,x,y0,cond,Ncut=3,npts=3,tol=1e-5,nMax=100)
    f0s = np.array([f(y0,x0) for y0,x0 in zip(y0s,x)])

    #plot
    y,cs = np.linspace(0,2,1000),dsp.getCs('jet',x.size)
    plts = [[y,f(y,x0),c] for x0,c in zip(x,cs)]
    plts += [[y0s[i],f0s[i],[cs[i],'s']] for i in range(x.size)]
    plts += [[[0,y.max()],[0,0],'k']]
    dsp.stddisp(plts,labs=['$y$','$f(y;x)$'],lw=2,xylims=['y',-20,20],ms=5)



def test_newton_cplx():
    f = lambda z,z0 : np.sqrt(z**2 - z0**2) #2x1
    df = lambda z,z0 : z/f(z,z0)    #2x2
    F  = lambda z,z0 : f_cplx(f,z,z0)
    dF = lambda z,z0 : j_cplx(df,z,z0)

    x,y0 = np.array([0,3]),[0,0]
    x0 = [z.real,z.imag]
    newton_rec(n,f,df,x,y0,Ncut=3,npts=3,tol=1e-5,nMax=100)
    x = sci_opt.newton(f,1.0+0.16J,df,args=(1+0.1J,));print(x)

def test_cplx_conjugation():
    f = lambda z,z0 : np.sqrt(z**2 - z0**2)
    df = lambda z,z0 : z/f(z,z0)
    z,z0 = 1.1+0.2J,   1+0.1J
    print('f(z)+f(zbar)=', (f(z,z0)+f(z.conjugate(),z0))/2  )
    print('f(z)-f(zbar)=', (f(z,z0)-f(z.conjugate(),z0))/2J )

######################################################################

# test_cplx_conjugation()
# x = nwt.newton(F,dF,x0,args=(z0,),tol=1e-5)
test_newton_rec()
#test_newton_cplx()
