from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


data = np.arange(100).reshape(10,10)

fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111,              # similar to subplot(111)
                 nrows_ncols = (3, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,          # pad between axes in inch.
                )

for i in range(12):
    grid[i].imshow(data) # The AxesGrid object work as a list of axes.  

plt.show()
