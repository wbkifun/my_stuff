#!/usr/bin/env python

import numpy as np
from matplotlib.pyplot import *

def func(x,y):
	return (1 - x/2 + x**5 + y**3)*np.exp(-x**2-y**2)

dx, dy = 0.05, 0.05

x = np.arange(-3.0, 3.0001, dx)
y = np.arange(-3.0, 3.0001, dx)
X,Y = np.meshgrid(x,y)

Z = func(X,Y)
imshow(Z)
colorbar()
#axis([-3,3,-3,3])

show()
