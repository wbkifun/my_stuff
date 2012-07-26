#!/usr/bin/env python

import scipy as sc
from pylab import *
from scipy.io.numpyio import fwrite
import pycuda.autoinit
import pycuda.driver as cuda


class FiniteDiff:
	def __init__( self, Nx, Ny, dx ):
		self.Nx = Nx
		self.Ny = Ny
		self.dx = dx


	def allocate( self ):
		self.A = sc.zeros( (self.Nx, self.Ny), 'f' )
		self.dA = sc.zeros_like( self.A )


	def allocate_in_dev( self ):
		self.dev_A = cuda.mem_alloc( self.A.nbytes )
		self.dev_dA = cuda.mem_alloc( self.A.nbytes )

	
	def free_in_dev( self ):
		self.dev_A.free()
		self.dev_dA.free()


	def get_kernels( self ):
		fpath = './CUDA/cudafunc.cu'
		mod = cuda.SourceModule( file( fpath,'r' ).reock=Db, grid=Dg )

_ad() )
		self.initmem = mod.get_function("initmem")
		self.diff = mod.get_function("diff")


	def initmem_in_dev( self ):
		Ntot = self.Nx*self.Ny

		tpb = 512
		if ( Ntot%tpb == 0 ): bpg = Ntot/tpb
		else: bpg = Ntot/tpb + 1

		Db = ( tpb, 1, 1 )
		Dg = ( bpg, 1 )

		self.initmem( sc.int32(Ntot), self.dev_A, block=Db, grid=Dg )
		self.initmem( sc.int32(Ntot), self.dev_dA, block=Db, grid=Dg )

_
	def initialize( self ):
		for i in xrange( self.Nx ):
			for j in xrange( self.Ny ):
				self.A[i,j] = sc.exp( - ( (i-self.Nx/2)*self.dx )**2/1.5 - ( (j-self.Ny/2)*self.dx )**2 )

		cuda.memcpy_htod( self.dev_A, self.A )


	def differentiate( self ):
		Ntot = (self.Nx - 1)*(self.Ny - 1)

		tpb = 512
		if ( Ntot%tpb == 0 ): bpg = Ntot/tpb
		else: bpg = Ntot/tpb + 1

		Db = ( tpb, 1, 1 )
		Dg = ( bpg, 1 )

		self.diff( sc.int32(self.Nx), sc.int32(self.Ny), sc.float32(self.dx), self.dev_A, self.dev_dA, block=Db, grid=Dg )


	def savefile( self, path ):
		cuda.memcpy_dtoh( self.dA, self.dev_dA )

		path1 = path + '-A_binary.dat'
		path2 = path + '-dA_binary.dat'

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
	obj.allocate_in_dev()
	obj.get_kernels()
	obj.initmem_in_dev()
	obj.initialize()
	obj.differentiate()
	obj.savefile( './dat/070' )

	obj.free_in_dev()
