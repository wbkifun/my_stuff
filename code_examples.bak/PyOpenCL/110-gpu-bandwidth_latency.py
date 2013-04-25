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
min_mbyte = 1
#max_mbyte = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) / (1024**2) / 2
max_mbyte = 128
num_samples = 128
print('Target gmem size: %d ~ %d MBytes' % (min_mbyte, max_mbyte))
print('Number of samples: %d' % num_samples)

#a = 4
#x = np.linspace(0, 1, num_samples)
#nbytes = np.int32(1/(np.e**a - 1) * ((max_mbyte - min_mbyte)*np.exp(a*x) + np.e**a * min_mbyte - max_mbyte) * 1024**2)
x = np.arange(1, num_samples + 1)
nxs = x[:] * Gs * 1024
#nbytes = x * Gs * 1024
etimes = np.zeros_like(x)

a_rand = np.random.rand(Gs).astype(np.float32)

# Main loop
mf = cl.mem_flags
for i, nx in enumerate(nxs):
	a = a_rand[:nx]
	b = np.zeros_like(a)
	a_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_gpu = cl.Buffer(context, mf.WRITE_ONLY, hostbuf=a)

	evt = prg.copy_gpu(queue, a.shape, None, np.int32(nx), a_gpu, b_gpu)
	evt.wait()
	etimes[i] = (evt.profile.END - evt.profile.START) * 1e-9 / 2

	cl.enqueue_read_buffer(queue, b_gpu, b)
	print np.linalg.norm(a - b)

	del a
	a_gpu.release()
	b_gpu.release()

	#print('%d/%d (%d %%)\r' % (i, num_samples, float(i)/num_samples*100)),
	#sys.stdout.flush()
	print('%d/%d (%d %%) time = %g, nbyte = %d, nx = %d' % (i, num_samples, float(i)/num_samples*100, etimes[i], nbyte, nx))

print('')

# Plot
import matplotlib.pyplot as plt
plt.ion()
plt.plot(etimes)
plt.show()
