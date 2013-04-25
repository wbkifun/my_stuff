#!/usr/bin/env python

import sys
import pyopencl as cl
import numpy as np
from my_cl_utils import print_device_info, get_optimal_global_work_size


# Platform, Device, Context and Queue
devices = []
platforms = cl.get_platforms()
for platform in platforms:
	devices.extend(platform.get_devices())
#print_device_info(platforms, devices)

device = devices[0]
context = cl.Context((device,))
queue = cl.CommandQueue(context, device, cl.command_queue_properties.PROFILING_ENABLE)


# Program and Kernel
Ls = 256
Gs = get_optimal_global_work_size(device)
print('Ls = %d, Gs = %d' % (Ls, Gs))

kernels = '''
__kernel void copy_gpu(const int nx, __global const float* a, __global float* b) {
	int idx = get_global_id(0);

	while( idx < nx ) {
		b[idx] = a[idx];
		idx += get_global_size(0);
	}
}'''
prg = cl.Program(context, kernels).build()


# Parameter setup
gmem_nbytes = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
nx0 = 512**2	# 1 MB = 1024**2 Byte = 512**2 float
nx1 = gmem_nbytes * 0.9 / 4 / 2	# / float bytes / num of arrays
#nx1 = 100*1024**2 * 0.9 / 4 / 2	# / float bytes / num of arrays
num_samples = 128
memcpy_iterations = 100
print('Target nx: %d ~ %d (%d ~ %d MBytes)' % (nx0, nx1, nx0*4/(1024**2), nx1*4/(1024**2)))
print('Number of samples: %d' % num_samples)

a = 4
x = np.linspace(0, 1, num_samples)
# exp distribution
#y = np.int32( 1/(np.e**a - 1) * ((nx1 - nx0)*np.exp(a*x[:]) + np.e**a * nx0 - nx1) )
# linear distribution
y = np.int32( (nx1 - nx0)*x[:] + nx0 )

seed_nxs = np.zeros(num_samples, dtype=np.int64)
elapsed_times = np.zeros(num_samples, dtype=np.float64)
print y.shape, y.dtype
for i in xrange(num_samples):
	mod = y[i] % 32
	if mod < 16:
		seed_nxs[i] = y[i] - mod
	else:
		seed_nxs[i] = y[i] + (32 - mod)

	assert seed_nxs[i] % 32 == 0
	

print('random generation: %d' % seed_nxs[-1])
path = './in_rand_%d.npy' % seed_nxs[-1]
try:
	in_rand = np.load(path)
except IOError:
	in_rand = np.random.rand(seed_nxs[-1]).astype(np.float32)
	np.save(path, in_rand)
print('in_rand: %s, %s' % (in_rand.shape, in_rand.dtype))


# Main loop
print('main loop')
from datetime import datetime
mf = cl.mem_flags
for i, nx in enumerate(seed_nxs):
	in_host = in_rand[:nx]
	out_host = np.zeros_like(in_host)
	in_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=in_host)
	out_gpu = cl.Buffer(context, mf.WRITE_ONLY, hostbuf=in_host)

	queue.finish()
	t0 = datetime.now()
	for j in xrange(memcpy_iterations):
		#prg.copy_gpu(queue, a.shape, None, np.int32(nx), in_gpu, out_gpu)
		cl.enqueue_copy_buffer(queue, in_gpu, out_gpu)
	queue.finish()
	dt = datetime.now() - t0
	elapsed_times[i] = (dt.seconds + dt.microseconds * 1e-6) / 2 / memcpy_iterations

	cl.enqueue_read_buffer(queue, out_gpu, out_host)
	assert np.linalg.norm(in_host - out_host) == 0

	del in_host, out_host
	in_gpu.release()
	out_gpu.release()

	#print('%d/%d (%d %%)\r' % (i, num_samples, float(i)/num_samples*100)),
	#sys.stdout.flush()
	print('%d/%d (%d %%) dt = %g sec, nx = %d (%d Mbytes)' % (i, num_samples, float(i)/num_samples*100, elapsed_times[i], nx, nx*4/(1024**2)))

print('')


# Save h5
import h5py as h5
gpu_name = device.get_info(cl.device_info.NAME)
h5_path = './measure_bandwidths.h5'
f = h5.File(h5_path, 'a')
gpu_groupid = 0
if 'gpu' not in f.keys():
	f.create_group('gpu')
if gpu_name not in f['gpu'].keys():
	f['gpu'].create_group(gpu_name)
	f['gpu'][gpu_name].attrs['gmem_nbytes'] = gmem_nbytes
f['gpu'][gpu_name].create_dataset('nx', data=seed_nxs)
f['gpu'][gpu_name].create_dataset('dt', data=elapsed_times)
f.close()


# Plot
import matplotlib.pyplot as plt
plt.ion()
plt.plot(elapsed_times)
plt.show()
