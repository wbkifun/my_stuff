#!/usr/bin/env python

import scipy as sc
from pylab import *

from fd_class import FiniteDiff


Nx, Ny = 400, 350
dx = 0.01

obj = FiniteDiff( Nx, Ny, dx )
obj.allocate()
obj.initialize()
obj.differentiate()


figure( figsize=(15,5) )
subplot(1,2,1)
imshow( transpose(obj.A), cmap=cm.hot, origin='lower', interpolation='bilinear' )
title('2D Gaussian')
xlabel('x-axis')
ylabel('y-axis')
colorbar()

subplot(1,2,2)
imshow( obj.dA.T, cmap=cm.hot, origin='lower', interpolation='bilinear' )
title('differential')
xlabel('x-axis')
ylabel('y-axis')
colorbar()

savefig('./png/040.png')
show()
