#!/usr/bin/env python

import scipy as sc
from pylab import *


Nx, Ny = 400, 350
dx = 0.01

A = sc.zeros( (Nx,Ny), 'f' )
dA = sc.zeros_like( A )

for i in xrange( Nx ):
	for j in xrange( Ny ):
		A[i,j] = sc.exp( - ( (i-Nx/2)*dx )**2/1.5 - ( (j-Ny/2)*dx )**2 )


dA[:-1,:-1] = (1/dx)*( A[1:,1:] - A[:-1,:-1] )


figure( figsize=(15,5) )
subplot(1,2,1)
imshow( transpose(A), cmap=cm.hot, origin='lower', interpolation='bilinear' )
title('2D Gaussian')
xlabel('x-axis')
ylabel('y-axis')
colorbar()

subplot(1,2,2)
imshow( dA.T, cmap=cm.jet, origin='lower', interpolation='bilinear' )
title('differential')
xlabel('x-axis')
ylabel('y-axis')
colorbar()

savefig('./png/020-fd.png')
show()
