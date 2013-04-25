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
print_device_info(platforms, devices)

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


# Launch
nx = Gs
mf = cl.mem_flags
a = np.random.rand(nx).astype(np.float32)
b = np.zeros_like(a)

a_gpu = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_gpu = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)

prg.copy_gpu(queue, a.shape, None, np.int32(nx), a_gpu, b_gpu)
cl.enqueue_read_buffer(queue, b_gpu, b)
print np.linalg.norm(a - b)

"""
	cl.wait_for_events([evt,])
	etimes[i] = (evt.profile.end - evt.profile.start) / 2

	del a
	a_gpu.release()
	b_gpu.release()

	#print('%d/%d (%d %%)\r' % (i, num_samples, float(i)/num_samples*100)),
	#sys.stdout.flush()
	print('%d/%d (%d %%) time = %d, nbyte = %d, nx = %d' % (i, num_samples, float(i)/num_samples*100, etimes[i], nbyte, nx))

print('')

# Plot
import matplotlib.pyplot as plt
plt.ion()
plt.plot(etimes)
plt.show()
"""
