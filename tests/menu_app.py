from utils import*
import menus;imp.reload(menus)
imp.reload(dsp)
plt.close('all')

dsp.matplotlib.rcParams['toolbar'] = 'None'
fig,ax = dsp.create_fig(figsize='12',f_args={'num':'ED tools'})
ax.axis('off')#plt.figure()
# fig.subplots_adjust(left=0.3)
props = menus.ItemProperties(labelcolor='black', bgcolor='yellow',
                       fontsize=15, alpha=0.2)
hoverprops = menus.ItemProperties(labelcolor='white', bgcolor='blue',
                            fontsize=15, alpha=0.2)

# menuitems = []
menuitems = {}
for label in ('open', 'close', 'save', 'save as', 'quit'):
    def on_select(item):
        print('you selected %s' % item.labelstr)
    # item = menus.MenuItem(fig, label, props=props, hoverprops=hoverprops,
    #                 on_select=on_select)
    # menuitems.append(item)
    menuitems[label] = on_select

y0=fig.get_figheight()*dsp.dpi[0]*0.5
menu = menus.Menu(fig, 'file',menuitems,props=props, hoverprops=hoverprops,x0=0,y0=y0)#y0)
# menu = menus.DropDownMenu(fig, menuitems,x0=0,y0=fig.get_figheight()*dsp.dpi[0]*0.9)
fig.show()
