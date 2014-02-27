from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

npts = 5000
m = Basemap(lon_0=270, boundinglat=20, projection='npstere')

# create randomly distributed points in map projection coordinates
x = uniform(m.xmin,m.xmax,npts)
y = uniform(m.ymin,m.ymax,npts)
xscaled = 4.*(x-0.5*(m.xmax-m.xmin))/m.xmax
yscaled = 4.*(y-0.5*(m.ymax-m.ymin))/m.ymax

# z is the data to plot at those points.
z = xscaled*np.exp(-xscaled**2-yscaled**2)
CS = plt.hexbin(x,y,C=z,gridsize=50,cmap=plt.cm.jet)

m.drawcoastlines()
m.drawparallels(np.arange(0,81,20))
m.drawmeridians(np.arange(-180,181,60))
m.colorbar() # draw colorbar

plt.show()
