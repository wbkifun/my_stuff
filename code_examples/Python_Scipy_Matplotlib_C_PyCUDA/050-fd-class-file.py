#!/usr/bin/env python

import scipy as sc
from pylab import *
from scipy.io.numpyio import fwrite


class FiniteDiff:
	def __init__( self, Nx, Ny, dx ):
		self.Nx = Nx
		self.Ny = Ny
		self.dx = dx


	def allocate( self ):
		self.A = sc.zeros( (self.Nx, self.Ny), 'f' )
		self.dA = sc.zeros_like( self.A )


	def initialize( self ):
		for i in xrange( self.Nx ):
			for j in xrange( self.Ny ):
				self.A[i,j] = sc.exp( - ( (i-self.Nx/2)*self.dx )**2/1.5 - ( (j-self.Ny/2)*self.dx )**2 )


	def differentiate( self ):
		self.dA[1:,1:] = (1/self.dx)*( self.A[1:,1:] - self.A[:-1,:-1] )


	def savefile( self ):
		fd = open('./dat/050-A_binary.dat', 'wb')
		fwrite( fd, self.A.size, self.A )
		fd.close()

		fd = open('./dat/050-dA_binary.dat', 'wb')
		fwrite( fd, self.dA.size, self.dA )
		fd.close()



if __name__ == '__main__':
	Nx, Ny = 400, 350
	dx = 0.01

	obj = FiniteDiff( Nx, Ny, dx )
	obj.allocate()
	obj.initialize()
	obj.differentiate()
	obj.savefile()
