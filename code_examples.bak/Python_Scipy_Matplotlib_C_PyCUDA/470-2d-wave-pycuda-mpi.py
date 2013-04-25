#!/usr/bin/env python

import scipy as sc
from matplotlib.pyplot import *
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import boostmpi as mpi

nbof = np.nbytes['float32']	# nbytes of float


class Wave2D:
	def __init__(s, nx, ny):
		s.nx, s.ny = nx, ny
		s._mem_allocate()


	def _mem_allocate(s):
		s.c = sc.ones((s.nx,s.ny),'f')*0.25
		s.f = sc.zeros_like(s.c)
		s.c_gpu = cuda.to_device(s.c)
		s.f_gpu = cuda.to_device(s.f)
		s.g_gpu = cuda.to_device(s.f)


	def prepare_kernel(s):
		mod = SourceModule("""
			__global__ void update_src(int idx, int tstep, float *f) {
				f[idx] += sin(0.1*tstep);
			}
			__global__ void update(int nx, int ny, float *c, float *f, float *g) {
				int tx = threadIdx.x;
				int idx = blockIdx.x*blockDim.x + tx;

				extern __shared__ float gs[];
				gs[tx+1] = g[idx];

				int i = idx/ny, j = idx%ny;
				if(j>0 && j<ny-1) {
					if(tx==0) gs[tx]=g[idx-1];
					if(tx==blockDim.x-1) gs[tx+2]=g[idx+1];
				}
				__syncthreads();

				if(i>0 && j>0 && i<nx-1 && j<ny-1) {
					f[idx] = c[idx]*(g[idx+ny]+g[idx-ny]+gs[tx+2]+gs[tx]-4*gs[tx+1])+2*gs[tx+1]-f[idx];
				}
			}
			""")
		s.update_src = mod.get_function("update_src")
		s.update = mod.get_function("update")

		Db, s.Dg = (256,1,1), (s.nx*s.ny/256+1, 1)
		s.nnx, s.nny = sc.int32(s.nx), sc.int32(s.ny)

		s.update_src.prepare("iiP", block=(1,1,1))
		s.update.prepare("iiPPP", block=Db, shared=(256+2)*4)

		
	def send(s, rank, tag_mark, arr_gpu):
		if mpi.rank > rank: offset_gpu = int(arr_gpu)+s.ny*nbof
		elif mpi.rank < rank: offset_gpu = int(arr_gpu)+(s.nx-2)*s.ny*nbof
		mpi.world.send(rank, tag_mark, cuda.from_device(offset_gpu, (s.ny,), s.f.dtype))


	def recv(s, rank, tag_mark, arr_gpu):
		if mpi.rank > rank: offset_gpu = int(arr_gpu)
		elif mpi.rank < rank: offset_gpu = int(arr_gpu)+(s.nx-1)*s.ny*nbof
		cuda.memcpy_htod(offset_gpu, mpi.world.recv(rank, tag_mark))

	
	def exchange(s, arr_gpu):
		if mpi.rank == 0:
			s.send(1, 0, arr_gpu)
			s.recv(1, 1, arr_gpu)
		if mpi.rank == 1:
			s.recv(0, 0, arr_gpu)
			s.send(0, 1, arr_gpu)
			s.send(2, 0, arr_gpu)
			s.recv(2, 1, arr_gpu)
		if mpi.rank == 2:
			s.recv(1, 0, arr_gpu)
			s.send(1, 1, arr_gpu)


	def send_output(s,rank,tag_mark):
		cuda.memcpy_dtoh(s.f,s.f_gpu)
		mpi.world.send(rank,tag_mark,s.f[1:-1,1:-1])



if __name__ == '__main__':
	# init the cuda
	cuda.init()
	ngpu = cuda.Device.count()
	ctx = cuda.Device(mpi.rank).make_context()

	# craete the space object
	nx_list = (1002,1002,1002)
	ny = 3002
	S = Wave2D(nx_list[mpi.rank],ny)
	S.prepare_kernel()

	# init the geometry
	if mpi.rank == 2:
		x0, y0 = 1, 1501
		dx = 20				# slit thickness
		w, d = 30, 170		# slit width, distance

		slx = slice(x0, x0+dx)
		sly1 = slice(y0-(d+w)/2, y0+(d+w)/2)
		sly2 = slice(y0-(d-w)/2, y0+(d-w)/2)
		S.c[slx,:] = 0
		S.c[slx,sly1] = 0.25
		S.c[slx,sly2] = 0

		cuda.memcpy_htod(S.c_gpu,S.c)

	# prepare for plot
	if mpi.rank == 0:
		output = sc.zeros((3000,3000),'f')

		ion()
		from matplotlib.patches import Rectangle
		rect1 = Rectangle((100,0),20,150,facecolor='0.4')
		rect2 = Rectangle((100,180),20,140,facecolor='0.4')
		rect3 = Rectangle((100,350),20,150,facecolor='0.4')
		gca().add_patch(rect1)
		gca().add_patch(rect2)
		gca().add_patch(rect3)
		#imsh = imshow(sc.ones((500,500),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
		imsh = imshow(output.T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
		colorbar()

	# main loop
	for tn in xrange(3000):
		if mpi.rank == 1: 
			src_pt = sc.int32(100*ny + 1501)
			S.update_src.prepared_call((1,1), src_pt, sc.int32(tn), S.f_gpu)

		S.update.prepared_call(S.Dg, sc.int32(S.nx), sc.int32(ny), S.c_gpu, S.f_gpu, S.g_gpu)
		S.exchange(S.f_gpu)

		S.update.prepared_call(S.Dg, sc.int32(S.nx), sc.int32(ny), S.c_gpu, S.g_gpu, S.f_gpu)
		S.exchange(S.g_gpu)

		if tn>100 and tn%100 == 0:
			if mpi.rank == 0:
				cuda.memcpy_dtoh(S.f,S.f_gpu)
				output[:1000,:] = S.f[1:-1,1:-1]
				output[1000:2000,:] = mpi.world.recv(1,10)
				output[2000:3000,:] = mpi.world.recv(2,10)
				#imsh.set_array( sc.sqrt(output[1900:2400,1250:1750].T**2) )
				imsh.set_array( sc.sqrt(output.T**2) )
				draw()
				#savefig('./png-wave/%.5d.png' % tstep) 
			else:
				S.send_output(0,10)
