from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



x = np.arange(-5, 5, 0.1)
y = np.arange(-4, 4, 0.1)
xx, yy = np.meshgrid(x, y)  # return shape (Ny,Nx)
#z = np.sin(xx**2 + yy**2)/(xx**2 + yy**2)
z = np.exp(-((xx-1)**2/(2*1.2**2) + (yy)**2/(2*1.5**2)))
print 'shape', xx.shape, yy.shape, z.shape

plt.contourf(x,y,z)
plt.show(True)
