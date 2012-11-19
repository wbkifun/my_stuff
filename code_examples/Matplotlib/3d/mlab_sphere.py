from __future__ import division
import numpy
from numpy import pi, linspace, sin, cos
from mayavi import mlab


R = 5
'''
npi = 10

lamda = linspace(-pi/2, pi/2, npi)
theta = linspace(-pi/4, pi/4, npi)
'''

dpi = pi/10
lamda, theta = numpy.mgrid[-pi/2:pi/2:dpi,-pi/4:pi/4:dpi]

x = R * cos(theta) * cos(lamda)
y = R * cos(theta) * sin(lamda)
z = R * sin(theta)

mlab.mesh(x, y, z, colormap='bone')
mlab.show()
