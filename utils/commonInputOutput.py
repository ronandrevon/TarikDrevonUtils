import numpy as np
import warnings
#import __builtin__
#defaultPath = 'C:\Users\ronan\Documents\Silvaco\PhysicsCode\Common'

#### def : Main user interface 
# load and save arrays
def saveParams(name,dat, datType):
    path = checkPath()
    filepath = '%s/dat/%s_%s' %(path,name,datType)
    np.save(filepath,dat)
    print('%s_%s successfully saved in %s' %(name,datType, path))
    
def loadParams(name,datType):
    print(' Loading %s_%s ' %(name,datType))
    path = checkPath()
    filepath = '%s/dat/%s_%s.npy' %(path,name,datType)
    dat = np.load(filepath)
    return dat

def printParams(paramNames,params,f='f',prec=3,s=5):
    d  = {'f':0,'E':1}[f]
    xp = lambda x:[
        ['%.0f' %x,'%.1f' %x,'%.2f' %x,'%.3f' %x,'%.4f' %x ,'%.5f' %x,
         '%.6f' %x,'%.7f' %x,'%.8f' %x,'%.9f' %x,'%.10f' %x],
        ['%.0E' %x,'%.1E' %x,'%.2E' %x,'%.3E' %x,'%.4E' %x ,'%.5E' %x,
         '%.6E' %x,'%.7E' %x,'%.8E' %x,'%.9E' %x,'%.10E' %x]][d][prec]
    xs = lambda x:[
        '%-0s' %x,'%-1s' %x,'%-2s'%x,'%-3s'%x,'%-4s'%x,'%-5s'%x,'%-6s'%x,'%-7s'%x,'%-8s'%x,'%-9s'%x,
        '%-10s'%x,'%-11s'%x,'%-12s'%x,'%-13s'%x,'%-14s'%x,'%-15s'%x,'%-16s'%x,'%-17s'%x,'%-18s'%x,'%-19s'%x,
        '%-20s'%x][s]  
    
    for paramUnit,p in zip(paramNames,params):
        paramName,unit = getParamUnit(paramUnit)
        pName = xs(paramName)
        if isinstance(p,int):
            print('%s = %d %s'          %(pName,p, unit))
        elif isinstance(p,float):
            print('%s = %s %s'          %(pName,xp(p),unit))
        elif isinstance(p,complex):
            print('%s = %s + j%s %s'    %(pName,xp(p.real),xp(p.imag),unit))

def getParamUnit(paramUnit):
    paramSplit = paramUnit.split('(')
    param = paramSplit[0]
    unit  = paramSplit[1].split(')')[0] if len(paramSplit)>1 else ''
    return param,unit

def printMatrices(Mat,MatNames,prec=2):
    np.set_printoptions(precision=prec)
    np.set_printoptions(suppress=False)
    for i in range(len(Mat)):
        print('%s : ' %MatNames[i])
        print(Mat[i])

#### handle complex for optimization
cplx2list = lambda c:[c.real,c.imag]
list2cplx = lambda c:c[0]+1j*c[1]

#### Misc :
# def checkPath():
    # if not hasattr(__builtin__, "path"):
        # warnings.warn( "Variable 'path' not defined. default path used instead :"
                       # "\n %s" %defaultPath)
        # path = defaultPath.replace('\\' ,'/').replace('\r' ,'/r').replace('\t' ,'/t')
    # else:
        # path = __builtin__.path
    # return path

def initVector(M,N):
    V = [np.zeros((N)) for i in range(M)]
    return V


#################################################################################
#### def : test
#################################################################################
def testArraySaveLoadPrint():
    name ='test'
    dat = np.random.rand(5)
    saveParams(name,dat,'array')
    datLoaded = loadParams(name,'array')
    params = ['a%d' %i for i in range(len(dat))]
    printParams(params,dat)

def testPrintParams():
    print('\t prec=3,s=2,f="E"')
    printParams(['i','f','c'],[1,2.0,1+1j],prec=3,s=2,f='E')
    print('\t prec=2,s=8')
    printParams(['integer','float','complex'],[1,2.0,1+1j],prec=2,s=8)
    print('\t prec=2,s=8')
    printParams(['test','gain(cm-1)','E(meV)','T(K)'],[1,100,1,0.0,300])
    
def testgetParamUnit():
    paramUnits = ['param','param(unit)','param(unit0']
    for paramUnit in paramUnits:
        print(getParamUnit(paramUnit))
    
if __name__=='__main__':
    #testArraySaveLoadPrint()
    #testPrintParams()
    #testgetParamUnit()
    print('InputOutput defined')
    print("ok")
