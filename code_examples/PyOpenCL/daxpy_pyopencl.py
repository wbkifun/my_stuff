#b------------------------------------------------------------------------------
# filename  : daxpy_pyopencl.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.8.22     start
#------------------------------------------------------------------------------

import numpy as np
import pyopencl as cl
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal


# kernel compile and import
kernel = '''
__kernel void daxpy(int nx, double a, __global double *x, __global double *y) {
    int idx = get_global_id(0);
    if (idx < nx) y[idx] = a*x[idx] + y[idx];
}
'''

platform_number = 0     # Intel
device_number = 0       # Xeon CPU

platforms = cl.get_platforms()
devices = platforms[platform_number].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context, devices[device_number])

print('platforms', platforms)
print('devices', devices)


prg = cl.Program(context, kernel)
#mod = prg.build(cache_dir='./')

prg.build()
binary = prg.get_info(cl.program_info.BINARIES)[0]
with open('daxpy.clbin', 'wb') as f: f.write(binary)
with open('daxpy.clbin', 'rb') as f:
    binary = f.read()
    binaries = [binary for device in devices]
    prg2 = cl.Program(context, devices, binaries)
    mod = prg2.build()


# setup
nx = 2**20

# allocation
a = np.random.rand()
x = np.random.rand(nx)
y = np.random.rand(nx)
y1 = np.zeros_like(y)
y2 = np.zeros_like(y)

# allocation on device
mf = cl.mem_flags
nx_cl = np.int32(nx)
a_cl = np.float64(a)
x_dev = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
y_dev = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)

# run
mod.daxpy(queue, (nx,), None, nx_cl, a_cl, x_dev, y_dev)

# check
y1[:] = a*x + y
cl.enqueue_copy(queue, y2, y_dev)
a_equal(y1, y2)
#aa_equal(y1, y2, 15)
print(y2-y1)
