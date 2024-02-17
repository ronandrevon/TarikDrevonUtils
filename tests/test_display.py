from utils import*
import utils.displayStandards as dsp;imp.reload(dsp)
from utils import pytest_util
import pytest,matplotlib


# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_stddisp_quick():
    return dsp.stddisp([range(2),range(2),'b'],title='test',
        labs=['$x$','$y$'],xylims=3,
        axPos=[0.12, 0.12, 0.8, 0.8], c=['k','b'],
        fonts={'text':45,'leg':35,'lab':25,'title':50}, texts=[[0.05,0.2,'O','m']],
        xyTicks=2,xyTicksm=0.5,#xyTickLabels=,
        pOpt='tXgGp',legLoc='center left',
        opt='')

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_3d():
    x,y,z = np.random.rand(3,25)
    xs,ys = np.meshgrid(range(10),range(10))
    zs = np.exp(-(xs**2+ys**2))
    dsp.stddisp(scat=[x,y,z,50,'b'],opt='')
    return dsp.stddisp(rc='3d',
        scat=[x,y,z,50,'b'],
        texts=[[0,0,0,'O','r']],
        xylims=2,
        surfs=[[xs,ys,zs,'b',0.25,2,None],[xs,ys,zs,{'alpha':0.5}]],
        legLoc='upper left',
        opt='')

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_inset():
    x = np.linspace(0,10,1000)
    plts=[x,np.exp(x),'b']
    inset={'axpos':[0.12,0.12,0.25,0.25],'xylims':[0,1,1,2],'opt':''}
    return dsp.stddisp(plts,xylims=10,axPos=[0.10,0.10,0.85,0.85],inset=inset,
        opt='')


# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_stddisp_im():
    x,y = np.meshgrid(range(250),range(250))
    z = np.exp(-(x**2+y**2))
    dsp.stddisp(im=[z],pOpt='im',opt='')
    dsp.stddisp(im=[x,y,z],pOpt='im',opt='sc',name='out/im.png')
    return dsp.stddisp(im='out/im.png',pOpt='',
        opt='')

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_image_bg():
    plts=[range(2),range(2),'b']
    dsp.stddisp(plts,axPos='F',xylims=1,opt='sc',name='out/bg.png')
    fig,ax=dsp.stddisp(opt='')
    dsp.image_bg('out/bg.png',rot=90,xylims=1,plots=plts,fig=fig,ax=ax,opt='')
    dsp.fix_pos(fig)
    return fig,ax


# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_stddisp_multi_plot():
    xi = [np.linspace(a,b,10) for a,b in zip([-2,-1,-3],[2]*3)]
    cs = dsp.getCs('jet',len(xi))
    plots = [[x, x**i, cs[i],'$x_%d$' %i] for i,x in enumerate(xi)]
    return dsp.stddisp(plots,opt='')

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_stddisp_multi_ax():
    fig,ax = dsp.create_fig(rc='12',figsize='f')
    x = np.linspace(0,1,10)
    dsp.stddisp([x,2*x,'b','$y1$'],['$x$','$y$'], ax=ax[0], opt='')
    dsp.stddisp([x,4*x,'r','$y2$'],['$t$','$y$'], ax=ax[1], opt='')
    return fig,ax[0]

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_AddyAxis():
    nPts = 10
    x1   = np.linspace(-2,2,nPts);y1 = pow(x1,1)
    plotsAx1 = [x1,+y1,'r' ,'']
    plotsAx2 = [x1,-y1,'b','']
    fig,ax = dsp.stddisp(plotsAx1,['$x$','$y$'],c=['k','r'],legOpt=0,opt='')
    dsp.addyAxis(fig,ax,plotsAx2,'$-y$',c='b', opt='')
    return fig,ax

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_AddxAxis():
    x1 = np.linspace(-2,2,100);y1 = pow(x1,2)
    x2 = np.linspace(-1,1,100);y2 = pow(x2,2)
    plotsAx1 = [x1,y1,'r' ,'$x^2$']
    plotsAx2 = [x2,y2,'b' ,'$x^2$']
    fig,ax = dsp.stddisp(plotsAx1,['$x$','$y$'],c=['k','k'],legOpt=0,opt='')
    dsp.addxAxis(ax,plotsAx2,'$x_2$',c='b', opt='')
    return fig,ax


###############################################################################
###############################################################################
# @pytest.mark.lvl1
def test_font():
    for s in '123':print(dsp.get_font(s))

# @pytest.mark.lvl1
def test_ChangeAxLim():
    fig,ax = dsp.stddisp([range(2),range(2),'b'],opt='')
    print(dsp.changeAxesLim(ax,0.05,xylims=[]))
    print(dsp.changeAxesLim(ax,0.05,xylims=[0,1,0,2]))
    print(dsp.changeAxesLim(ax,0.05,xylims=[0,1]))
    print(dsp.changeAxesLim(ax,0.05,xylims=['x',0,1]))
    print(dsp.changeAxesLim(ax,0.05,xylims=['y',0,2]))
    print(dsp.changeAxesLim(ax,0.05,xylims=0.2))

# @pytest.mark.lvl1
def test_get_axPos():
    for key in dsp.axPos.keys():
        if isinstance(key,str):key="'%s'" %key
        cmd ="dsp.get_axPos(%s)" %key
        print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))
    for key in [0,[],[0.1, 0],[0.1, 0.1, 0.8, 0.8]]:
        cmd ='dsp.get_axPos(%s)' %str(key) ;
        print('%-20s: axPos=%s' %(cmd,str(eval(cmd))))


# def test_GetSymCurve():
#     x = np.linspace(0,np.pi,100)
#     x1,x2 = x,x
#     y1,y2 = np.cos(x1),np.sin(x2)
#     x1,y1 = getSymCurve(x1,y1,1)
#     x2,y2 = getSymCurve(x2,y2,-1)
#     stdDispPlt([[x1,y1,'b','$cos$'],[x2,y2,'r','$sin$']])
# @pytest.mark.lvl1
def test_GetCML():
    cml_tests=['b','b-o',(0,1,1),[(0,1,1),'o'],[(0,1,1),'-']]
    for s in cml_tests:
        c,m,l = dsp.getCML(s)
        print('color, marker, linestyle')
        print(c,m,l)

# @pytest.mark.lvl1
def test_scat() :
    x,y = np.random.rand(2,10)
    c   = ['b']*10
    s   = [5]*10
    ct  = [(0.2,0.3,0.1)]*10
    dsp.stddisp(scat=[x,y,s,c]          ,opt='q')
    dsp.stddisp(scat=[x,y,s[0],c[0]]    ,opt='q')
    dsp.stddisp(scat=[x,y,c]            ,opt='q')
    dsp.stddisp(scat=[x,y,np.array(c)]  ,opt='q')
    dsp.stddisp(scat=[x,y,ct]           ,opt='q')
    dsp.stddisp(scat=[x,y,ct[0]]        ,opt='q')
    dsp.stddisp(scat=([x,y,ct[0],'o'],[x,2*y,ct[0],'x']),opt='q')

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_polygon():
    pol  = matplotlib.patches.Polygon([[0,0],[0,1],[1,1]],label='$pol$')
    coll = matplotlib.collections.PatchCollection([pol],'y',alpha=0.3,linewidth=2,edgecolor='g')
    legElt={'pol':pol}
    return dsp.stddisp(colls=[coll],opt='')#,legElt=legElt,)

# @pytest.mark.lvl1
@pytest_util.add_link(__file__)
def test_display_solutions():
    disp_d = {
        'x1':('get_x1','b',r'$x_1$'),
        'x2':('get_x2','c',r'$x_2$'),
        'y1':('get_y1','r',r'$y_1$'),
        'y2':('get_y2','m',r'$y_2$'),
        }
    class C1:
        def get_x1(self):return np.linspace(0,1,10)
        def get_x2(self):return np.linspace(0,2,10)
        def get_y1(self):return 1*self.get_x1()
        def get_y2(self):return 2*self.get_x2()
    class C2:
        def get_x1(self):return np.linspace(0,1.5,20)
        def get_x2(self):return np.linspace(0,3.0,20)
        def get_y1(self):return 0.5*self.get_x1()
        def get_y2(self):return 3*self.get_x2()
    c1,c2 = C1(),C2()
    objs=[(c1,'ko','$c_1$'), (c2,'ks','$c_2$')]

    dsp.display_solutions(disp_d,objs,key_pair=('x1','y1'),help=0,lw=2,opt='')
    fig,ax,labs,legElt=dsp.display_solutions(disp_d,objs,key_pair=('x2',['y1','y2']),help=0,lw=2,
        opt='')
    return fig,ax

if __name__=='__main__':
    plt.close('all')
    fig,ax=test_inset();fig.show()
    #_testGetCML('b:.<')
    #_testStdDispPlt()
    #_testStdDispPlt_b()
    #_testStdDispPlt_c()
    #_testStdDispPlt_subplots()
    #_testAddxAxis()
    #_testAddyAxis()
    #_testChangeAxLim()
    #_testGetSymCurve()
    #_test_get_axPos()
    #_test_scat()
    #_test_stddisp()
    #print(getCML(np.random.rand(3))
    # test_display_solutions()
