from __future__ import division
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# lon_0, lat_0 are the center point of the projection.
# resolution = 'l' means use low resolution coastlines.
bm = Basemap(projection='ortho', lon_0=127.5, lat_0=38, resolution='l')
bm.drawcoastlines()
bm.fillcontinents(color='coral',lake_color='aqua')

# draw parallels and meridians.
bm.drawparallels(numpy.arange(-90.,120.,30.))
bm.drawmeridians(numpy.arange(0.,420.,60.))
bm.drawmapboundary(fill_color='aqua')

plt.title("Full Disk Orthographic Projection")
plt.show()
