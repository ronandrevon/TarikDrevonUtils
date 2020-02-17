import matplotlib;
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
from .glob_colors import*
###########################################################################
#### def: plotting standards 
def standardDisplay(ax,labs,name='', xylims=[], axPos=[],legOpt=1,
                    fonts=[30,25,15,20], c=['k','k'], logOpt='',changeXYlims=True,
                    gridOn=True,ticksOn=True,title='',legLoc='upper left',legElt=[],
                    xyTicks=[],xyTickLabs=[[],[]],opt='',setPos=True):
    axPos = get_axPos(axPos); #print(axPos)
    ax.name = name; 
    if isinstance(fonts,dict) : fonts = get_font(fonts,dOpt=True)
    fs,fsLeg,fsL,fsT = fonts; #print(fsLeg)
    #xy axis and labels
    if 'x' in logOpt : ax.set_xscale('log')
    if 'y' in logOpt : ax.set_yscale('log')
    ax.set_xlabel(labs[0],fontsize=fs   ,color=c[0])
    ax.set_ylabel(labs[1],fontsize=fs   ,color=c[1])
    if len(labs)==3 : ax.set_zlabel(labs[2],fontsize=fs   ,color=c[1])
    ax.tick_params('x',labelsize=fsL    ,colors=c[0])
    ax.tick_params('y',labelsize=fsL    ,colors=c[1])
    #position,grid,title,legend,save
    if setPos : ax.set_position(axPos)
    if gridOn : ax.grid(gridOn,color=(0.9,0.9,0.9),linestyle='-')
    ax.set_axisbelow(True)
    if changeXYlims : changeAxesLim(ax,0.05, xylims,xyTicks,xyTickLabs)
    plt.axis(['off','on'][ticksOn])
    ax.set_title(title, {'fontsize': fsT}); #print(title)
    addLegend(ax,fsLeg,legOpt,legLoc,legElt)
    disp_quick(name,ax,opt)

def stdDispPlt(plots,xlabs=['',''],name='',xlims=[],axPos=[],c=['k','k'],
               showOpt=1,ax=None,fig=None,texts=[],legOpt=1,lw=1,ms=5,fonts={},
               logOpt='',changeXYlims=True,gridOn=True,ticksOn=True,fullsize=False,title='',
               legLoc='upper left',legElt=[],xyTicks=[],colls=[],opt='',figsize=(9,9)):
    '''fonts.keys=['lab','leg','tick','text','title']'''
    if isinstance(fonts,dict) : fonts = get_font(fonts)
    fsT = fonts[3]; fonts=(np.array(fonts)[[0,1,2,4]]).tolist()
    if showOpt and not 'q' in opt : opt+='p'
    if not ax : fig,ax = plt.subplots(figsize=figsize)
    if ax and axPos==[] : axPos=ax.get_position().bounds
    if fullsize : mng = plt.get_current_fig_manager(); mng.resize(*mng.window.maxsize())
    for coll in colls : ax.add_collection(coll)
    pltPlots(ax,plots,lw,ms)
    pltTexts(ax,texts,fsT)
    standardDisplay(ax,xlabs,name,xlims,axPos ,legOpt,fonts,c,
                    logOpt=logOpt,changeXYlims=changeXYlims,gridOn=gridOn,title=title,
                    legLoc=legLoc,legElt=legElt,xyTicks=xyTicks,opt=opt)
    return fig,ax

def addyAxis(fig,ax,plots,yLab='',c='k', lw=1,ms=5,axPos=[],showOpt=1,yTicks=[],yTickLabs=[],
             **kwargs):
    axPosI = axPos if len(axPos)==4  else [0.13, 0.15, 0.75, 0.75]
    ax2 = ax.twinx()
    pltPlots(ax2,plots,lw)
    standardDisplay(ax2,['',yLab],c=['k',c],axPos=axPosI,xyTicks=[[],yTicks],xyTickLabs=[[],yTickLabs],
                    **kwargs)
    ax.set_position(axPosI)
    if showOpt : plt.show()

def addxAxis(ax,plots,xLab='',c='k', lw=1,axPos=[],
            xTicks=[],xTickLabs=[], **kwargs):
    axPosI = axPos if len(axPos)==4  else [0.13, 0.15, 0.8, 0.675]
    ax2 = ax.twiny()
    pltPlots(ax2,plots,lw)
    standardDisplay(ax2,[xLab,''],c=[c,'k'],axPos=axPosI,xyTicks=[xTicks,[]],xyTickLabs=[xTickLabs,[]],**kwargs)
    ax2.tick_params('x',color=c)
    ax2.set_xlabel(xLab, verticalalignment='bottom')
    ax.set_position(axPosI)
    if showOpt : plt.show()

def get_font(d_font=dict(),dOpt=False) : 
    keys = ['lab','leg','tick','text','title']
    vals = [30,25,15,20,30]
    font_dict = dict(zip(keys,vals))
    for k in d_font.keys() : font_dict[k] = d_font[k]
    if dOpt : keys = ['lab','leg','tick','title']
    fonts = [ font_dict[k] for k in keys]
    return  fonts

#### def : utils 
def pltPlots(ax,plots,lw0,ms0=5):
    if len(plots)==4:
        plots = [plots] if isinstance(plots[3],str) else plots
    for p in plots:
        c,m,l = getCML(p[2])
        lw = lw0 if len(p)<5 else p[4]
        ax.plot(p[0],p[1],label=p[3],color=c,linestyle=l,marker=m, linewidth=lw,markersize=ms0)

def pltTexts(ax,texts,fsT):
    if any(texts):
        texts = [texts] if not isinstance(texts[0],list) else texts
    for t in texts:
        c = 'k' if len(t)<4 else t[3]
        ax.text(t[0],t[1],t[2],fontsize=fsT,color=c)
        
def getCML(C):
    ml = ''
    if isinstance(C,list):
        c,ml = C[0], C[1]
    elif isinstance(C,tuple) or isinstance(C,np.ndarray):
        c,m,l = C, '','-'
    elif isinstance(C,str):
        c,m,l,ml = C[0],'','-',C[1:]
    else:
        c,m,l= 'b','','-'
        
    if not ml == '':
        l = [char for char in ml if char in '-.:']    ;l=''.join(l)
        m = [char for char in ml if char not in '-.:'];m=''.join(m)
    return c,m,l

def addLegend(ax,fsLeg,legOpt,loc='upper left',legElt=[]):
    if legOpt:
        if any(legElt):
            leg = ax.legend(handles=legElt,fontsize=fsLeg,loc=loc)
        else:
            leg = ax.legend(fontsize=fsLeg,loc=loc)
    if legOpt:
        leg.set_draggable(True)         # matplotlib 3.1
        #leg.draggable(True)            # matplotlib 2.2
        #if leg : leg.DraggableLegend() # matplotlib?

def getCs(name,N):
    cmap = matplotlib.cm.get_cmap(name)
    cs = [cmap(float(i+1)/N)[0:3] for i in range(N)]
    return cs

def changeAxesLim(ax,mg,xylims=[],xyTicks=[],xyTickLabs=[[],[]]):
    data = ax.dataLim.bounds
    xm,ym,W,H = data
    xM = xm+W; yM = ym+H
    xmin = xm-mg*W; xmax = xM+mg*W
    ymin = ym-mg*H; ymax = yM+mg*H
    #change limits
    if   len(xylims)==4:
        xmin,xmax,ymin,ymax = xylims
    elif len(xylims)==3:
        if xylims[0]=='x':
            xmin,xmax = xylims[1:3]
        if xylims[0]=='y':
            ymin,ymax = xylims[1:3]
    if   len(xylims)==6:
        xmin,xmax,ymin,ymax,zmin,zmax = xylims
        ax.set_zlim((zmin, zmax));#print('ok')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    
    xylims = [xmin, xmax,ymin, ymax]
    # change ticks
    if any(xyTicks) :
        dx,dy = xyTicks
        lab_x,lab_y = xyTickLabs;#print(lab_x,lab_y)
        if any(lab_x):
            plt.xticks(np.arange(xmin,xmax,dx),lab_x)
        else:
            plt.xticks(np.arange(xmin,xmax,dx))
        if any(lab_y):
            plt.yticks(np.arange(ymin,ymax,dy),lab_y)
        else:
            plt.yticks(np.arange(ymin,ymax,dy))
    return xylims

# def add_cb():
    # # ax.axis(extent);ax.grid(True);ax.legend(loc='lower left');
    # # sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Greys',ns));sm.set_array([]);
    # # cb=fig.colorbar(sm,boundaries=0.5+np.arange(ns+1),ticks=range(1,ns+1));cb.ax.set_yticklabels(['%d' %(n) for n in Ns])

def get_axPos(axPosI):    
    axPos = {'bigTitle':[0.15,0.2,0.75,0.6],
        1:[0.13, 0.15, 0.75, 0.75],
        11:[0.1, 0.1, 0.35, 0.8],
        12:[0.6, 0.1, 0.35, 0.8],
        21:[0.07, 0.1, 0.9, 0.4],
        22:[0.07, 0.6, 0.9, 0.4],
        31:[0.15, 0.18, 0.75, 0.75]}
    axPosition = axPos[1]
    if isinstance(axPosI,int):
        if axPosI in axPos.keys():
            axPosition = axPos[axPosI]
    elif isinstance(axPosI,list) or isinstance(axPosI,tuple):
        if len(axPosI)==4:
            axPosition = axPosI
    # elif isinstance(axPosI,matplotlib.transforms.Bbox):
        # axPosition = axPosI
    return axPosition

def saveFig(path, ax,png=None,fmt='png',opt='p'):
    if not png==None : fmt=['eps','png'][png]
    filename = path+'.'+fmt
    plt.savefig(filename, format=fmt, dpi=96)
    print(green+'figure '+yellow+filename+green+' succesfully saved'+black)

def disp_quick(name,ax,opt):
    if 's' in opt : saveFig(name, ax,fmt='png')
    if 'p' in opt : plt.show()
    if 'c' in opt : plt.close()
#Template for series of plots 
''' 
def stdMultDisp(X,Ys,cs=[],labs=[],iDs=[],xlabs=['',''],name='',xlims=[],axPos=[],showOpt=1):       
    N     = len(Ys)
    cs    = [ (0,(i+1)/N,0) for i in range(N)    ] if not any(cs)   else cs
    labs  = [ '' for i in range(N)               ] if not any(labs) else labs   
    plots = [[E,Ys[i],cs[i],'$G_{%s}$' %(labs[i])] for i in range(N) ]
    stdDispPlt(plots,xlabs,name,xlims,axPos,showOpt)
'''




###########################################################################
#### defs : printing stuffs
def valsToStr(v,n=2,f='E'):
    if f == 'E':
        if n==1:
            vals = ['%.1E' %(val) for val in v];
        elif n==2:
            vals = ['%.2E' %(val) for val in v];
        else:
            vals = ['%.3E' %(val) for val in v]
    elif f=='f' :
        if n==1:
            vals = ['%.1f' %(val) for val in v];
        elif n==2:
            vals = ['%.2f' %(val) for val in v];
        else:
            vals = ['%.3f' %(val) for val in v]
    elif f=='d' :
        vals = ['%d' %(val) for val in v];
    return  '[' + ', '.join(vals) + ']'

def printVals(v,f='E',prec=3):
    values = ''
    d  = {'f':0,'E':1}[f]
    xp = lambda x:[
        ['%.0f' %x,'%.1f' %x,'%.2f' %x,'%.3f' %x,'%.4f' %x ,'%.5f' %x,
         '%.6f' %x,'%.7f' %x,'%.8f' %x,'%.9f' %x,'%.10f' %x],
        ['%.0E' %x,'%.1E' %x,'%.2E' %x,'%.3E' %x,'%.4E' %x ,'%.5E' %x,
         '%.6E' %x,'%.7E' %x,'%.8E' %x,'%.9E' %x,'%.10E' %x]][d][prec]
    for val in v:
        if isinstance(val,int):                        
            values += '%-12d' %(val) 
        elif isinstance(val,str):
            values += '%-12s' %(val)
        else:            
            values += '%-12s' %(xp(val))
    print(values)
    
def printKeys(k):
    keys, line = ['','']
    strLine = '-'*12
    for key in k:
        keys   = keys + '%-12s' %(key) 
        line   = line + strLine      
    print(keys)
    print(line)
    
def printList(k,v):
    keys ,values, line = ['','','']
    strLine = '-'*10
    for key,val in zip(k,v):
        keyStr = '%-10s' %(key) if val >0 else '%-11s' %(key)
        keys   = keys + keyStr
        line   = line + strLine
        values = values + '%.2E' %(val) + '  '        
    print(keys)
    print(line)
    print(values)


###########################################################################
# Class : event Handler generic
class eventHandlerStd:
    def __init__(self,fig,ax,path):
        self.cid  = fig.canvas.mpl_connect('key_press_event', self)
        self.fig  = fig
        self.ax   = ax
        self.RS = RectangleSelector(
            ax, self.line_select_callback, drawtype='box', useblit=True,
            button=[1, 3],spancoords='pixels', minspanx=20,minspany=20 )
        self.path = path

        self.selectBox = [0.0,0.0,0.0,0.0]
        self.params = {'paramTest':[0,1]}
        plt.rcParams['keymap.save'] = ''
    def __call__(self, event):
        #print ('key entered: ' + key) 
        if event.key=='s':
            self.save()
        if event.key=='l':
            self.load()
            
        if event.key == '0':
            self.action0()
        if event.key in ['1','2','3','4','5','6','7','8','9']:
            self.displayAx(int(event.key))
        if event.key == 'enter':
            self.returnAction()
        if event.key == 'backspace':
            self.backspaceAction()
        if event.key == ' ':
            self.spaceAction()
        if event.key in ['up','right','left','down']:
            self.arrowActions(event.key)
        if event.key in ['a','q']:
            self.activateBox(event.key)
        if event.key in 'zertyuiopdfghjkmwxcvbn':
            self.keyCallBack(event.key)
        self.update()
        
    #can be overriden
    def save(self):
        saveFig(self.path, self.ax)
    def load(self):
        print('Base load ')
    def update(self):
        print('Base update')
    
    def keyCallBack(self,key):
        print(key)
    def activateBox(self,key):
        if key == 'q':
            self.activateRS(0)
        if key == 'a':
            self.activateRS(1)                
    def activateRS(self,b):
        if b==0:
            print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
        if b==1:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)
            
    def line_select_callback(self,eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.selectBox = [x1,x2,y1,y2] 
        print(self.selectBox)

    def displayAx(self,i):
        print('base display ' + str(i))
    def action0(self):
        print('base action0')
    def returnAction(self):
        print('base returnAction')
    def backspaceAction(self):
        print('base backspaceAction')
    def spaceAction(self):
        print('base spaceAction')
    def arrowActions(self,key):
        print('base arrowAction ' + key)
        
def arrowActionsParam(v,dv,key):
    if key == 'up':
        v = v+dv
    if key == 'down':
        v = v-dv
        v = max(v,dv)
    if key == 'right':
        dv = 2*dv
    if key == 'left':
        dv = 0.5*dv
    return v,dv




###########################################################################
#### def : handy
def getSymCurve(x,y,symOpt=1):
    x = np.concatenate((np.flipud(-x),x))
    y = np.concatenate((np.flipud(symOpt*y),y))
    return x,y



###########################################################################
##### defs : Tests
###########################################################################
def testHandler():
    fig,ax = plt.subplots()
    x = np.linspace(0,1,100)
    ehd=eventHandler(fig,ax,'',x)
    plt.show()
def testPrint():
    l = ['mue(meV)', 'muh(meV)', 'Ne(/cm^3)', 'Nh(/cm^3)', 'G(/cm)']
    printKeys(l)
    printVals([0.5,-0.8, 1.0*pow(10.,19),5.0*pow(10.,16), -250.023])
    printVals([-0.5,-0.8, 1.0*pow(10.,19),+5.0*pow(10.,16), -250.023])
    printVals([0.5,0.8, -1.0*pow(10.,18),5.0*pow(10.,16), 250.023])
def testValsToStr(): 
    v = [1.25254,3.2545,5.245,5.02]
    print(valsToStr(v,1,'E'))
    print(valsToStr(v,2))
    print(valsToStr(v,3,''))

def testGetCML(s):
    c,m,l = getCML(s)
    print('color, marker, linestyle')
    print(c,m,l)

################################ display tests
def testStdDispPlt():
    nPts = 10
    x1 = np.linspace(-2,2,nPts);y1 = pow(x1,1)
    x2 = np.linspace(-2,2,nPts);y2 = pow(x2,2)
    x3 = np.linspace(-2,2,nPts);y3 = pow(x3,3)
    plots = [[x1,y1,'r+-.','$x$'  ,1],
             [x2,y2,'gs--','$x^2$',2],
             [x3,y3,'bo--','$x^3$',3]]
    xlabs = ['$x$','$y$']
    stdDispPlt(plots,xlabs, axPos=[0.12, 0.12, 0.8, 0.8], c=['k','b'],
               fonts=[30,25,12,25], texts=[[0.05,0.2,'O','m']])
def testStdDispPlt_b():
    x = np.linspace(0,1,10)
    plots       = [x,x,'b','x']
    plotsList   = [plots]
    stdDispPlt(plots, showOpt=0)
    stdDispPlt(plotsList)
def testStdDispPlt_c():
    x = np.linspace(0,1,10)
    plots       = [x,x,[(1.0,0.5,0.2),'s-.'],'x']
    stdDispPlt(plots, showOpt=1)
def testStdDispPlt_subplots():
    fig,ax = plt.subplots(nrows=1,ncols=2)
    x = np.linspace(0,1,10)
    stdDispPlt([x,2*x,'b','$y1$'],['$x$','$y$'], ax=ax[0],axPos=11, showOpt=0)
    stdDispPlt([x,4*x,'r','$y2$'],['$t$','$y$'], ax=ax[1],axPos=12, showOpt=1,fullsize=True)
    
def testAddyAxis():
    nPts = 10
    x1   = np.linspace(-2,2,nPts);y1 = pow(x1,1)
    plotsAx1 = [x1,+y1,'r' ,'']
    plotsAx2 = [x1,-y1,'b','']
    fig,ax = stdDispPlt(plotsAx1,['$x$','$y$'],c=['k','r'],legOpt=0,showOpt=0)
    addyAxis(fig,ax,plotsAx2,'$-y$',c='b', legOpt=0, showOpt=1)

def testAddxAxis():
    x1 = np.linspace(-2,2,100);y1 = pow(x1,2)
    x2 = np.linspace(-1,1,100);y2 = pow(x2,2)
    plotsAx1 = [x1,y1,'r' ,'$x^2$']
    plotsAx2 = [x2,y2,'b' ,'$x^2$']
    fig,ax = stdDispPlt(plotsAx1,['$x$','$y$'],c=['k','k'],legOpt=0,showOpt=0)
    addxAxis(ax,plotsAx2,'$x_2$',c='b', showOpt=1)

def testChangeAxLim():
    fig,ax = plt.subplots();plt.plot(range(5),range(5))
    print(changeAxesLim(ax,0.05,xylims=[]))
    print(changeAxesLim(ax,0.05,xylims=[0,1.,0,2.]))
    print(changeAxesLim(ax,0.05,xylims=['x',0,1.]))
    print(changeAxesLim(ax,0.05,xylims=['y',0,2.]))

def test_get_axPos():
    cmd ='get_axPos(1)' ; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos(11)'; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos(21)'; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos(0)' ; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos([])'; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos([0.1, 0])'; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos([0.1, 0.1, 0.8, 0.8])'; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    cmd ='get_axPos((0.1, 0.1, 0.8, 0.8))'; print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))

def testGetSymCurve():
    x = np.linspace(0,np.pi,100)
    x1,x2 = x,x
    y1,y2 = np.cos(x1),np.sin(x2)
    x1,y1 = getSymCurve(x1,y1,1)
    x2,y2 = getSymCurve(x2,y2,-1)
    stdDispPlt([[x1,y1,'b','$cos$'],[x2,y2,'r','$sin$']])

if __name__=='__main__':
    #testHandler()
    #testPrint()
    #testValsToStr()
    #testGetCML('b:.<')
    #testStdDispPlt()
    #testStdDispPlt_b()
    #testStdDispPlt_c()
    #testStdDispPlt_subplots()
    #testAddxAxis()
    #testAddyAxis()    
    #testChangeAxLim()
    #print getCML(np.random.rand(3))
    #testGetSymCurve()
    #test_get_axPos()
    print('displayStandards compiled')
