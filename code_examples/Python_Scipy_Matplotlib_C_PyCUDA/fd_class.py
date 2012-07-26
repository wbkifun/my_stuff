#!/usr/bin/env python

import scipy as sc
from pylab import *


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
