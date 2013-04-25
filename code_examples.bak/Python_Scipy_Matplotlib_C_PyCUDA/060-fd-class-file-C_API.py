#!/usr/bin/env python

import scipy as sc
from pylab import *
from scipy.io.numpyio import fwrite

import sys
sys.path.append( './C' )
from cfunc import diff


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
		diff( self.Nx, self.Ny, self.dx, self.A, self.dA )


	def savefile( self, path ):
		path1 = path + 'A_binary.dat'
		path2 = path + 'dA_binary.dat'

		fd = open( path1, 'wb' )
		fwrite( fd, self.A.size, self.A )
		fd.close()

		fd = open( path2, 'wb' )
		fwrite( fd, self.dA.size, self.dA )
		fd.close()



if __name__ == '__main__':
	Nx, Ny = 400, 350
	dx = 0.01

	obj = FiniteDiff( Nx, Ny, dx )
	obj.allocate()
	obj.initialize()
	obj.differentiate()
	obj.savefile( './dat/060' )
