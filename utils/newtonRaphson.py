import numpy as np
import matplotlib.pyplot as plt
from .displayStandards import*

#### def : Standard
def newtonRaphson(f,df,x0,tol=0.01,nMax=10):
    x,i = x0,0
    fx = np.linalg.norm(f(x))
    while fx>=tol and i<nMax:   
        dx = np.linalg.solve(df(x),f(x))
        x = np.subtract(x,dx)
        fx = np.linalg.norm(f(x))
        i = i+1
    return x,fx,i

def newtonRaphson1D(f,df,x0,tol=0.01,nMax=10):
    x,i  = x0,0
    while abs(f(x))>=tol and i<nMax : 
        x = x - f(x)/df(x) 
        i = i+1
    return x,f(x),i

############################################################
#### def : Efficient 
def newtonRaphsonEff(fdf,x0,tol=0.01,nMax=10):
    x,i = x0,0
    fx,dfx = fdf(x)
    fxN = np.linalg.norm(fx)
    while fxN>=tol and i<nMax:
        fx,dfx = fdf(x)
        dx = np.linalg.solve(dfx,fx)
        x = np.subtract(x,dx)
        fxN = np.linalg.norm(fx)
        i = i+1
    return x

def newtonRaphson1DEff(fdf,x0,tol=0.01,nMax=10):
    x,i  = x0,0
    fx,dfx = fdf(x)
    while abs(fx)>=tol and i<nMax:
        x = x - fx/dfx 
        i = i+1
        fx,dfx = fdf(x)
    return x

############################################################
#### def : Debug 
def newtonRaphsonDebug(f,df,x0,tol=0.01,nMax=10):
    x = x0
    fx = np.linalg.norm(f(x))
    print(fx);plt.plot(x,'o-');
    i = 0
    while fx>=tol and i<nMax:   
        dx = np.linalg.solve(df(x),f(x))
        x = np.subtract(x,dx)
        fx = np.linalg.norm(f(x))
        i = i+1
        print(fx);plt.plot(x,'o-',color=(1-fx/tol,0,0));
    return x,fx,i
def newtonRaphson1Ddebug(f,df,x0,X,tol=0.01,nMax=10,logOpt=0):
    fX      = np.log10(abs(f(X))) if logOpt else f(X) 
    plots   = [[X ,fX ,'b-','$f(x)  $']]

    dispIntermediateSol(plots,x0,f,'r',tol,logOpt)
    x,i = x0,0
    while abs(f(x))>=tol and i<nMax:
        x = x - f(x)/df(x) 
        i = i+1
        dispIntermediateSol(plots,x,f,'b',tol,logOpt)
    dispIntermediateSol(plots,x,f,'g',tol,logOpt)

    stdDispPlt(plots,['x','fx'],xlims=['y',-100.,100.])
    return x,f(x),i


############################################################
#### def : Misc
def dispIntermediateSol(plots,x,f,c,tol,logOpt=0):
    fx = f(x)
    if logOpt:
        logTol  = abs(np.log10(tol))
        logFx   = logTol if fx<=tol else abs(np.log10(abs(fx)))
        #c       = abs(pow(logFx/logTol,0.3))
        fx      = abs(logFx)
    print(x, fx)
    plots.append([x,fx,c+'o',''])
    return plots

############################################################
##### def : Tests
def newtonExample(debugOpt=0):
    print('\t')                         ;print('Newton Raphson ' )    
    f  = lambda x:[pow(x[0],2)-1,x[1]]  ;print("f(x)  = [x(0)^2-1,x(1)] ")
    df = lambda x:[[2*x[0]  ,0],
                   [0       ,1]]        ;print("df(x) = [[2*x(0)^2,0] \n"
                                           "         [0       ,1]]")
    x0 = [2,2]                          ;print("x0 = [2,2]")
    x = newtonRaphson(f,df, x0)         ;print("x  = " + valsToStr(x[0]))
    if debugOpt:
        X = np.meshgrid(np.linspace(0,2,10),np.linspace(0,2,10))
        fX = f0(X)
        plt.plot();plt.show()
def newtonExample1D(debugOpt=0,logOpt=0):
    x0 = [-2.,2.5]
    print('\t')                     ;print('Newton Raphson 1D ' )
    f  = lambda x:pow(x,2)-1        ;print('f  = x^2-1')
    df = lambda x:2*x               ;print('df = 2*x')
    x  = newtonRaphson1D(f,df,x0[0]);print('x0=%-3.2f => x =%.2f'  %(x0[0],x[0])) 
    x  = newtonRaphson1D(f,df,x0[1]);print('x0=%-3.2f => x =%.2f'  %(x0[1],x[0])) 

    if debugOpt:
        X = np.linspace(-3,3,1002)
        newtonRaphson1Ddebug(f,df,x0[1],X,tol=pow(10.,-12),nMax=10,logOpt=logOpt)

def newtonExample1DdebugQuick():
    f  = lambda x:pow(x,3)-1
    df = lambda x:3*pow(x,2)
    X  = np.linspace(-3,3,1000)
    x  = newtonRaphson1Ddebug(f,df,-10.,X,tol=pow(10.,-5),nMax=20,logOpt=0);
    print(x)
    
if __name__=='__main__':
    #newtonExample(0)
    #newtonExample1D(debugOpt=1,logOpt=0)
    newtonExample1DdebugQuick()
    print('succesful compilation')