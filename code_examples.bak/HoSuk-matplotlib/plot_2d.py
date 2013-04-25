#!/usr/bin/env python


# Read the h5 file
import numpy as np
import h5py as h5
import sys
try:
	fpath_data = sys.argv[1]
	fpath_geo = sys.argv[2]
except IndexError:
	print('Error: h5 file names required.')
	print('Usage: ./plot_2d.py data.h5 geo.h5')
	sys.exit()

f = h5.File(fpath_data, 'r')
data = f['Data'].value
f.close()
f = h5.File(fpath_geo, 'r')
geo = f['Data'].value
f.close()


# Plot
from matplotlib.pyplot import *
imshow(geo.T, origin='lower', cmap=cm.gray)
imshow(data.T, origin='lower', alpha=0.9)
xlabel('x')
ylabel('y')
colorbar()

nx, ny = data.shape
ticklabels = ['%1.1f' % t for t in np.linspace(-1, 1, 11)]
xticks(np.linspace(0, nx, 11), ticklabels)
yticks(np.linspace(0, ny, 11), ticklabels)
savefig('field.png', dpi=150)
show()
