from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys


#-----------------------------------------
# setup
#-----------------------------------------
dx, dy = 1200, 1000
#dx, dy = 1200, 1000//2


#-----------------------------------------
# read a binary file
#-----------------------------------------
try:
    fpath = sys.argv[1]
    fp = open(fpath, 'rb')
except IndexError:
    print 'Usage:'
    print '$ python plot_binary.py binary_file_name'
    sys.exit()

field = np.fromfile(fp, count=dx*dy, dtype=np.float64).reshape((dx,dy), order='F')


#-----------------------------------------
# plot
#-----------------------------------------
plt.ion()
plt.imshow(field.T, origin='lower', vmin=-0.2, vmax=0.2)
plt.colorbar()
plt.show(True)
