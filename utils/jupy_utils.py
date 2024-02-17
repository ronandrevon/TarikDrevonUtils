from IPython.display import display, Markdown, Latex
import numpy as np

def show_hkls(path,name,hkls,n=5):
    '''show figures from rocking curves '''
    rep = lambda s:s.replace(', ','_').replace('(','').replace(')','')
    pattern = lambda s:'%s/%s_%s.png' %(path,name,s)
    imageIDs = [rep(h) for h in hkls]
    show_figs(pattern,imageIDs,n=n)

def show_figs(pattern,imageIDs,n=5,v=0):
    '''
    display figures in table format.
    Parameters :
    -----------
    pattern  : pattern for the path to the images
    imageIDs : list of image ID (each file is obtained as 'pattern(ID)')
    n : number of images per rows

    Note :
        The IDs are also used as titles in the tables
    '''
    
    nrows=len(imageIDs)//n
    hs_=np.reshape(imageIDs[:n*nrows],(nrows,n))
    txt=''
    for hs in hs_:
        txt +='\n%s\n%s --\n%s\n'%(
            ' | ' .join(hs),
            ' -- | '*(len(hs)-1),
            ' | '.join([ '![](%s)' %pattern(h) for h in hs] ),
        )
    if v:print(txt)
    display(Markdown(txt))
