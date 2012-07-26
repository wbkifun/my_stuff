#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from my_cl_utils import print_device_info


# Platform, Device, Context and Queue
devices = []
queues = []
platforms = cl.get_platforms()
for platform in platforms:
	devices.extend(platform.get_devices())
#print_device_info(platforms, devices)

context = cl.Context(devices)
for device in devices:
	queues.append(cl.CommandQueue(context, device))

# Parameter setup
ngpu = len(devices)
snx = 256 * 1000000;	# sub_nx, about 2G
nx = snx * ngpu;
tmax = 100

mem_size = (snx * np.nbytes['float32'] * 2) / (1024**2)
print('host: %1.3f MBytes' % (mem_size*ngpu))
print('gpu: %1.3f MBytes' % mem_size)

# Program and Kernel
kernels = open('./vec_op.cl').read()
kern = kernels.replace('NX',str(nx))
program = cl.Program(context, kern).build()

# Allocation
a = np.zeros(nx, dtype=np.float32)
b = np.zeros(nx, dtype=np.float32)
a_init = np.random.rand(256).astype(np.float32)
b_init = np.random.rand(256).astype(np.float32)
for i in xrange(nx/256):
	a[i*256:(i+1)*256] = a_init[:]
	b[i*256:(i+1)*256] = b_init[:]

mf = cl.mem_flags
a_gpus = []
b_gpus = []
for i, queue in enumerate(queues):
	a_gpus.append(cl.Buffer(context, mf.READ_WRITE, snx*np.nbytes['float32']))
	b_gpus.append(cl.Buffer(context, mf.READ_ONLY, snx*np.nbytes['float32']))
	cl.enqueue_write_buffer(queue, a_gpus[i], a[i*snx:(i+1)*snx])
	cl.enqueue_write_buffer(queue, b_gpus[i], b[i*snx:(i+1)*snx])

# Kernel launch
for tstep in xrange(tmax):
	for queue, a_gpu, b_gpu in zip(queues, a_gpus, b_gpus):
		program.vecadd_gpu(queue, (snx,), (256,), a_gpu, b_gpu)
		program.vecsub_gpu(queue, (snx,), (256,), a_gpu, b_gpu)

# Verification
'''
for tstep in xrange(tmax):
	a[:] += b[:]
	a[:] -= b[:]
'''

for i, queue in enumerate(queues):
	cl.enqueue_read_buffer(queue, a_gpus[i], b[i*snx:(i+1)*snx])

#print np.linalg.norm(a - b)
