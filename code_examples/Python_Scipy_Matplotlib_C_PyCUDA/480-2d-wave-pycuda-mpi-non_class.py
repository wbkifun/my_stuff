#!/usr/bin/env python

import numpy as np
import pycuda.driver as cuda
import boostmpi as mpi

nbof = np.nbytes['float32']	# nbytes of float32
dtof = np.dtype('float32')	# dtype of float32 numpy array


kernels = """
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
"""


def send(rank, tag_mark, nx, ny, arr_gpu):
	if mpi.rank > rank: offset_gpu = int(arr_gpu)+(ny+1)*nbof
	elif mpi.rank < rank: offset_gpu = int(arr_gpu)+((nx-2)*ny+1)*nbof
	mpi.world.send(rank, tag_mark, cuda.from_device(offset_gpu, (ny-2,), dtof))


def recv(rank, tag_mark, nx, ny, arr_gpu):
	if mpi.rank > rank: offset_gpu = int(arr_gpu)+1*nbof
	elif mpi.rank < rank: offset_gpu = int(arr_gpu)+((nx-1)*ny+1)*nbof
	cuda.memcpy_htod(offset_gpu, mpi.world.recv(rank, tag_mark))


def exchange(nx, ny, arr_gpu):
	if mpi.rank == 0:
		send(1, 0, nx, ny, arr_gpu)
		recv(1, 1, nx, ny, arr_gpu)
	if mpi.rank == 1:
		recv(0, 0, nx, ny, arr_gpu)
		send(0, 1, nx, ny, arr_gpu)
		send(2, 0, nx, ny, arr_gpu)
		recv(2, 1, nx, ny, arr_gpu)
	if mpi.rank == 2:
		recv(1, 0, nx, ny, arr_gpu)
		send(1, 1, nx, ny, arr_gpu)



if __name__ == '__main__':
	# init the cuda
	cuda.init()
	ngpu = cuda.Device.count()
	ctx = cuda.Device(mpi.rank).make_context()

	# memory allocate
	nx, ny = 1002, 3002
	c = np.ones((nx,ny),'f')*0.25
	f = np.zeros_like(c)
	c_gpu = cuda.to_device(c)
	f_gpu = cuda.to_device(f)
	g_gpu = cuda.to_device(f)

	# set the geometry
	if mpi.rank == 2:
		x0, y0 = 100, 1501
		dx = 20				# slit thickness
		w, d = 30, 170		# slit width, distance

		slx = slice(x0, x0+dx)
		sly1 = slice(y0-(d+w)/2, y0+(d+w)/2)
		sly2 = slice(y0-(d-w)/2, y0+(d-w)/2)
		c[slx,:] = 0
		c[slx,sly1] = 0.25
		c[slx,sly2] = 0

		cuda.memcpy_htod(c_gpu, c)

	# prepare kernels
	from pycuda.compiler import SourceModule
	mod = SourceModule(kernels)
	update_src = mod.get_function("update_src")
	update = mod.get_function("update")

	Db, Dg = (256,1,1), (nx*ny/256+1, 1)
	nnx, nny = np.int32(nx), np.int32(ny)
	update_src.prepare("iiP", block=(1,1,1))
	update.prepare("iiPPP", block=Db, shared=(256+2)*4)
	if mpi.rank == 1: src_pt = np.int32(1100*ny + 1501)

	# prepare for plot
	if mpi.rank == 0:
		output = np.zeros((3000,3000),'f')

		from matplotlib.pyplot import *
		ion()
		imsh = imshow(output.T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
		'''
		from matplotlib.patches import Rectangle
		rect1 = Rectangle((200,0),20,150,facecolor='0.4')
		rect2 = Rectangle((200,180),20,140,facecolor='0.4')
		rect3 = Rectangle((200,350),20,150,facecolor='0.4')
		gca().add_patch(rect1)
		gca().add_patch(rect2)
		gca().add_patch(rect3)
		imsh = imshow(np.ones((500,500),'f').T, cmap=cm.hot, origin='lower', vmin=0, vmax=0.1)
		'''
		colorbar()


	#--------------------------------------------------------------------
	# main loop
	for tn in xrange(3000):
		if mpi.rank == 1: update_src.prepared_call((1,1), src_pt, np.int32(tn), f_gpu)
		update.prepared_call(Dg, nnx, nny, c_gpu, f_gpu, g_gpu)
		exchange(nx, ny, f_gpu)
		update.prepared_call(Dg, nnx, nny, c_gpu, g_gpu, f_gpu)
		exchange(nx, ny, g_gpu)

		if tn>1000 and tn%10 == 0:
			if mpi.rank == 0:
				cuda.memcpy_dtoh(f, f_gpu)
				output[:1000,:] = f[1:-1,1:-1]
				output[1000:2000,:] = mpi.world.recv(1,10)
				output[2000:3000,:] = mpi.world.recv(2,10)
				#imsh.set_array( np.sqrt(output[1900:2400,1250:1750].T**2) )
				imsh.set_array( np.sqrt(output.T**2) )
				draw()
				#savefig('./png-wave/%.5d.png' % tstep) 
			else:
				cuda.memcpy_dtoh(f, f_gpu)
				mpi.world.send(0, 10, f[1:-1,1:-1])
	
	ctx.pop()
