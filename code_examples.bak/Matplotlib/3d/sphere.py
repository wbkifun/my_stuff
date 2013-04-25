from __future__ import division
from mpl_toolkits.mplot3d import axes3d
import numpy
from numpy import linspace, pi, sin, cos, tan
import matplotlib.pyplot as plt


l = linspace(0, 2*pi, 100)       # lambda
t = linspace(-pi/2, pi/2, 100)   # theta

X = cos(t) * cos(l)
Y = cos(t) * sin(l)
Z = sin(t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
plt.show()

