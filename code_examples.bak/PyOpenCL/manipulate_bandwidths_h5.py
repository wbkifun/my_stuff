#!/usr/bin/env python

import numpy as np
import h5py as h5

path = './measure_bandwidths.h5'
f = h5.File(path, 'r')

# cpu
cpu_name = f['cpu'].keys()[0]
cpu0 = f['cpu'][cpu_name]
nx_cpu_copy = cpu0['nx_copy'].value
dt_cpu_copy = cpu0['dt_copy'].value
nx_cpu_write = cpu0['nx_write'].value
dt_cpu_write = cpu0['dt_write'].value
nx_cpu_read = cpu0['nx_read'].value
dt_cpu_read = cpu0['dt_read'].value
nbytes_cpu_copy = nx_cpu_copy[:] * 4.
nbytes_cpu_write = nx_cpu_write[:] * 4.
nbytes_cpu_read = nx_cpu_read[:] * 4.

# gpu
gpu_name = f['gpu'].keys()[0]
gpu0 = f['gpu'][gpu_name]
nx_gpu = gpu0['nx_memcpy'].value
dt_gpu = gpu0['dt_memcpy'].value
nbytes_gpu = nx_gpu[:] * 4.

items = ['cpu copy', 'cpu write', 'cpu read', 'gpu memcpy']
results = []
nbytes_list = [nbytes_cpu_copy, nbytes_cpu_write, nbytes_cpu_read, nbytes_gpu]
dt_list = [dt_cpu_copy, dt_cpu_write, dt_cpu_read, dt_gpu]

print('cpu name = %s' % cpu_name)
print('gpu name = %s' % gpu_name)
print('gmem_size = %d Bytes' % gpu0.attrs['gmem_nbytes'])
print('sample size = %s' % nx_gpu.shape)


# Fitting
from scipy import optimize
fitfunc = lambda p, x: p[0] * x + p[1]
errfunc = lambda p, x, y: fitfunc(p, x) - y

p0 = np.array([100*1e9, 0])
for nbytes, dt in zip(nbytes_list, dt_list):
	p1, success = optimize.leastsq(errfunc, p0, args=(nbytes, dt))
	#p1 = optimize.fmin_slsqp(errfunc, p0, args=(nbytes, dt))
	bandwidth = 1 / p1[0]
	latency = p1[1]
	results.append((bandwidth, latency))


# Print
for item, result in zip(items, results):
	print('%s: bandwidth = %g GB/s, latency = %g s' % (item, result[0]/1e9, result[1]))


# Plot
import matplotlib.pyplot as plt
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Memory Size (GB)')
ax1.set_ylabel('Time (ms)')

for nbytes, dt, result in zip(nbytes_list, dt_list, results):
	ax1.plot(nbytes/(1024**3), dt*1e3, linestyle='None', marker='p', markersize=2)

	x = np.array([nbytes[0], nbytes[-1]])
	y = np.array([1./result[0] * x[0] + result[1], 1./result[0] * x[1] + result[1]])
	ax1.plot(x/(1024**3), y*1e3)

plt.show()
