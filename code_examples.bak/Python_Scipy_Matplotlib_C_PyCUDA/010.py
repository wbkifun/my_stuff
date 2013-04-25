#!/usr/bin/env python

import scipy as sc
from pylab import *


Nx, Ny = 400, 350
dx = 0.01

A = sc.zeros( (Nx,Ny), 'f' )

for i in xrange( Nx ):
	for j in xrange( Ny ):
		A[i,j] = sc.exp( - ( (i-Nx/2)*dx )**2/1.5 - ( (j-Ny/2)*dx )**2 )


imshow( transpose(A), cmap=cm.hot, origin='lower', interpolation='bilinear' )
title('2D Gaussian')
xlabel('x-axis')
ylabel('y-axis')
colorbar()
savefig('./png/010.png')
show()
