from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm



#X, Y, Z = axes3d.get_test_data(0.05)
X = np.linspace(-10, 10, 200)
Y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(X, Y)
G = np.exp(-(X**2 + Y**2)/10)
Z = np.sin(G)


fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(1,1,1)

pcm = ax1.pcolormesh(X, Y, Z)
'''
ax1 = fig.add_subplot(1,1,1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax1.contour(X, Y, Z, zdir='z', offset=-1.5, cmap=cm.coolwarm)
cset = ax1.contour(X, Y, Z, zdir='x', offset=-10, cmap=cm.coolwarm)
cset = ax1.contour(X, Y, Z, zdir='y', offset=10, cmap=cm.coolwarm)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_zlim(-1.5, 1.5)

#surf.set_array(Z)
'''
#plt.show()


for tstep in xrange(1,100):
    Z = np.sin( tstep*0.01*G )
    pcm.set_array(Z)
    plt.draw()
