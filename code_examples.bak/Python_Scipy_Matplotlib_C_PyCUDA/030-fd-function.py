#!/usr/bin/env python

import scipy as sc
from pylab import *


def allocate( Nx, Ny ):
	A = sc.zeros( (Nx,Ny), 'f' )
	dA = sc.zeros_like( A )

	return A, dA


def initialize( Nx, Ny, dx, A ):
	for i in xrange( Nx ):
		for j in xrange( Ny ):
			A[i,j] = sc.exp( - ( (i-Nx/2)*dx )**2/1.5 - ( (j-Ny/2)*dx )**2 )


def differentiate( dx, A, dA ):
	dA[1:,1:] = (1/dx)*( A[1:,1:] - A[:-1,:-1] )



if __name__ == '__main__':
	Nx, Ny = 400, 350
	dx = 0.01

	A, dA = allocate( Nx, Ny )
	initialize( Nx, Ny, dx, A )
	differentiate( dx, A, dA )


	figure( figsize=(15,5) )
	subplot(1,2,1)
	imshow( transpose(A), cmap=cm.hot, origin='lower', interpolation='bilinear' )
	title('2D Gaussian')
	xlabel('x-axis')
	ylabel('y-axis')
	colorbar()

	subplot(1,2,2)
	imshow( dA.T, cmap=cm.hot, origin='lower', interpolation='bilinear' )
	title('differential')
	xlabel('x-axis')
	ylabel('y-axis')
	colorbar()

	savefig('./png/030.png')
	show()
