#!/usr/bin/env python

import sys
import numpy as np


# Parameter setup
gmem_nbytes = 1024**3
nx0 = 512**2	# 1 MB = 1024**2 Byte = 512**2 float
nx1 = gmem_nbytes * 0.9 / 4 / 2	# / float bytes / num of arrays
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

seed_nxs = np.zeros(num_samples, dtype=np.int32)
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


# Program and Kernel
import subprocess
from ctypes import c_int
kernels = open('memcpy.c').read()
kernels = kernels.replace('OMP_MAX_THREADS',str(4))
print kernels

of = open('/tmp/memcpy.c', 'w')
of.write(kernels)
of.close()
cmd = 'gcc -O3 -std=c99 -fpic -shared -fopenmp -msse %s -o /tmp/libmemcpy.so' %(of.name)
subprocess.Popen(cmd.split())
clib = np.ctypeslib.load_library('libmemcpy', '/tmp/')
clib.memread.restype = None


# Main loop
print('main loop')
from datetime import datetime
for i, nx in enumerate(seed_nxs):
	in_host = in_rand[:nx]

	arg = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, shape=(nx,), flags='C_CONTIGUOUS, ALIGNED')
	clib.memread.argtypes = [c_int, arg]

	t0 = datetime.now()
	for j in xrange(memcpy_iterations):
		clib.memread(nx, in_host)
	dt = datetime.now() - t0
	elapsed_times[i] = (dt.seconds + dt.microseconds * 1e-6) / memcpy_iterations

	del in_host

	print('%d/%d (%d %%) dt = %g sec, nx = %d (%d Mbytes)' % (i, num_samples, float(i)/num_samples*100, elapsed_times[i], nx, nx*4/(1024**2)))

print('')


# Save h5
import h5py as h5
for line in open('/proc/cpuinfo'):
	if 'model name' in line:
		cpu_name = line[line.find(':')+1:-1]
		break;
h5_path = './measure_bandwidths.h5'
f = h5.File(h5_path, 'a')
if 'cpu' not in f.keys():
	f.create_group('cpu')
if cpu_name not in f['cpu'].keys():
	f['cpu'].create_group(cpu_name)
f['cpu'][cpu_name].create_dataset('nx_read', data=seed_nxs)
f['cpu'][cpu_name].create_dataset('dt_read', data=elapsed_times)
f.close()


# Plot
import matplotlib.pyplot as plt
plt.ion()
plt.plot(elapsed_times)
plt.show()
