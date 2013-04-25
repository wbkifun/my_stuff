#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from my_cl_utils import print_device_info


# Platform, Device, Context and Queue
devices = []
platforms = cl.get_platforms()
for platform in platforms:
	devices.extend(platform.get_devices())
#print_device_info(platforms, devices)

device = devices[0]
context = cl.Context((device,))
queue = cl.CommandQueue(context, device)

# Parameter setup
nx = 256 * 10000;
tmax = 100

mem_size = (nx * np.nbytes['float32'] * 2) / (1024**2)
print('host: %1.3f MBytes' % mem_size)
print('gpu: %1.3f MBytes' % mem_size)

# Program and Kernel
kernels = open('./vec_op.cl').read()
program = cl.Program(context, kernels.replace('NX',str(nx))).build()
#vecadd_gpu = cl.Kernel(program, 'vecadd_gpu')
#vecsub_gpu = cl.Kernel(program, 'vecsub_gpu')

# Allocation
a = np.zeros(nx, dtype=np.float32)
b = np.zeros(nx, dtype=np.float32)
a_sub = np.random.rand(256).astype(np.float32)
b_sub = np.random.rand(256).astype(np.float32)
for i in xrange(nx/256):
	a[i*256:(i+1)*256] = a_sub[:]
	b[i*256:(i+1)*256] = b_sub[:]

mf = cl.mem_flags
a_gpu = cl.Buffer(context, mf.READ_WRITE, nx*np.nbytes['float32'])
b_gpu = cl.Buffer(context, mf.READ_ONLY, nx*np.nbytes['float32'])
cl.enqueue_write_buffer(queue, a_gpu, a)
cl.enqueue_write_buffer(queue, b_gpu, b)

# Kernel launch
#vecadd_gpu.set_args(a_gpu, b_gpu)
#vecsub_gpu.set_args(a_gpu, b_gpu)
kernel_args = (a_gpu, b_gpu)

for tstep in xrange(tmax):
	program.vecadd_gpu(queue, (nx,), (256,), *kernel_args)
	program.vecsub_gpu(queue, (nx,), (256,), *kernel_args)

# Verification
a0 = a[:]
for tstep in xrange(tmax):
	a[:] += b[:]
	a[:] -= b[:]
cl.enqueue_read_buffer(queue, a_gpu, b)
print np.linalg.norm(a - a0)
print np.linalg.norm(a - b)
