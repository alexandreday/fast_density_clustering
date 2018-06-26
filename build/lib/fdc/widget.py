import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class Highlighter(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.x, self.y = x, y
        self.mask = np.zeros(x.shape, dtype=bool)

        self._highlight = ax.scatter([], [], s=200, color='yellow', zorder=10)

        self.selector = RectangleSelector(ax, self, useblit=False)
        if self.selector.active:
            self.selector.update()
        plt.show()

    def __call__(self, event1, event2):
        self.mask |= self.inside(event1, event2)
        xy = np.column_stack([self.x[self.mask], self.y[self.mask]])
        self._highlight.set_offsets(xy)
        self.canvas.draw()

    def inside(self, event1, event2):
        """Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2."""
        # Note: Could use points_inside_poly, as well
        x0, x1 = sorted([event1.xdata, event2.xdata])
        y0, y1 = sorted([event1.ydata, event2.ydata])
        mask = ((self.x > x0) & (self.x < x1) &
                (self.y > y0) & (self.y < y1))
        return mask