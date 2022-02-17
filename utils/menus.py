import matplotlib.artist as artist
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.transforms import IdentityTransform


class ItemProperties:
    def __init__(self, fontsize=14, labelcolor='black', bgcolor='yellow',
                 alpha=1.0):
        self.fontsize = fontsize
        self.labelcolor = labelcolor
        self.bgcolor = bgcolor
        self.alpha = alpha


class MenuItem(artist.Artist):
    padx = 5
    pady = 5

    def __init__(self, fig, labelstr, props=None, hoverprops=None,
                 on_select=None,menu=None):
        super().__init__()

        self.set_figure(fig)
        self.menu=menu
        self.labelstr = labelstr

        self.props = props if props is not None else ItemProperties()
        self.hoverprops = (
            hoverprops if hoverprops is not None else ItemProperties())
        if self.props.fontsize != self.hoverprops.fontsize:
            raise NotImplementedError(
                'support for different font sizes not implemented')

        self.on_select = on_select

        # Setting the transform to IdentityTransform() lets us specify
        # coordinates directly in pixels.
        self.label = fig.text(0, 0, labelstr.ljust(20), transform=IdentityTransform(),
                              ha='left',va='bottom',size=props.fontsize)
        self.text_bbox = self.label.get_window_extent(
            fig.canvas.get_renderer())

        # self.rect = patches.Rectangle((0, 0), 1, 1)  # Will be updated later.

        self.set_hover_props(False)

        self.cid=fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        # over, _ = self.rect.contains(event)
        over, _ = self.label.contains(event)
        if not over:
            return
        if self.on_select is not None:
            self.on_select(self)

    def set_extent(self, x, y, w, h, depth):
        # self.rect.set(x=x, y=y, width=w, height=h)
        self.label.set(position=(x + self.padx, y + depth + self.pady/2))
        self.hover = False

    def draw(self, renderer):
        # self.rect.draw(renderer)
        self.label.draw(renderer)

    def set_hover_props(self, b):
        props = self.hoverprops if b else self.props
        self.label.set(color=props.labelcolor, backgroundcolor=props.bgcolor,alpha=props.alpha)
        # self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

    def set_hover(self, event):
        """
        Update the hover status of event and return whether it was changed.
        """
        b, _ = self.label.contains(event)
        changed = (b != self.hover)
        if changed:
            self.set_hover_props(b)
        self.hover = b
        return changed


class Menu:
    def __init__(self, fig,label,menu_items,props, hoverprops,x0=100,y0=400):
        self.figure = fig
        self.down=False
        self.y0=y0
        def on_select(self,):
            if not self.menu.down:
                print('showing DropDownMenu')
                menuitems = [MenuItem(fig, label, props=props, hoverprops=hoverprops,
                    on_select=on_select,menu=self.menu) for label,on_select in menu_items.items()]
                maxw = max(item.text_bbox.width for item in menuitems)
                maxh = max(item.text_bbox.height for item in menuitems)
                depth = max(-item.text_bbox.y0 for item in menuitems)

                width = maxw + 2*MenuItem.padx
                height = maxh + MenuItem.pady
                y0=self.menu.y0
                for item in menuitems:
                    left = x0
                    bottom = y0 - maxh - MenuItem.pady

                    item.set_extent(left, bottom, width, height, depth)

                    fig.artists.append(item)
                    y0 -= maxh + MenuItem.pady
                self.menu.menuitems=menuitems
                self.menu.down=True
                self.figure.canvas.draw()
            else:
                print('menu already down')

        props1       = ItemProperties(labelcolor='black', bgcolor=(0.5,)*3,fontsize=15, alpha=0.2)
        hoverprops1  = ItemProperties(labelcolor='white', bgcolor='blue',fontsize=15, alpha=0.2)
        item = MenuItem(fig, label, props=props1, hoverprops=hoverprops1,on_select=on_select,menu=self)
        item.set_extent(x0, y0, item.text_bbox.width+2*MenuItem.padx , item.text_bbox.height, -item.text_bbox.y0)
        # fig.artists.append(item.label)
        self.button=item
        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        if self.button.set_hover(event) and not self.down:
            self.figure.canvas.draw()
        elif self.down:
            if not any(item.hover for item in self.menuitems) and not self.button.hover:
                print('dropdown menu off')
                for item in self.menuitems :
                    self.figure.canvas.mpl_disconnect(item.cid)
                    self.figure.artists.remove(item)#self.figure.artists.remove(item.rect)
                    item.label.remove()#self.figure.artists.remove(item.rect)
                    self.down=False
                self.figure.canvas.draw()
            elif any(item.set_hover(event) for item in self.menuitems):
                self.figure.canvas.draw()



class DropDownMenu:
    def __init__(self, fig, menuitems,x0=100,y0=400):
        self.figure = fig
        self.menuitems = menuitems

        maxw = max(item.text_bbox.width for item in menuitems)
        maxh = max(item.text_bbox.height for item in menuitems)
        depth = max(-item.text_bbox.y0 for item in menuitems)

        width = maxw + 2*MenuItem.padx
        height = maxh + MenuItem.pady

        for item in menuitems:
            left = x0
            bottom = y0 - maxh - MenuItem.pady

            item.set_extent(left, bottom, width, height, depth)

            fig.artists.append(item.label)
            y0 -= maxh + MenuItem.pady

        fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_move(self, event):
        if any(item.set_hover(event) for item in self.menuitems):
            self.figure.canvas.draw()
