#b------------------------------------------------------------------------------
# filename  : apb.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.9.6      start
#------------------------------------------------------------------------------

import numpy as np
import pyopencl as cl
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal


# kernel compile and import
platform_number = 0     # Intel
device_number = 0       # Xeon CPU

platforms = cl.get_platforms()
platform = [p for p in platforms if 'Intel' in p.vendor][0]
devices = platform.get_devices()
device = [d for d in devices if cl.device_type.to_string(d.type)=='CPU'][0]

context = cl.Context([device])
queue = cl.CommandQueue(context, device)

print('platform', platform)
print('device', device)


'''
prg = cl.Program(context, kernel)
#mod = prg.build(cache_dir='./')

prg.build()
binary = prg.get_info(cl.program_info.BINARIES)[0]
with open('daxpy.clbin', 'wb') as f: f.write(binary)
'''
with open('build/apb.clbin', 'rb') as f:
    binary = f.read()
    binaries = [binary for device in devices]
    prg2 = cl.Program(context, devices, binaries)
    lib = prg2.build()


# setup
nx = 2**20

# allocation
nx = 1000000
a = np.random.rand(nx)
b = np.random.rand(nx)
c = np.random.rand(nx)

# ref value
kk, lll, mm = 2, 3.5, 1.7
ref = kk*a + lll*b + mm*c

# allocation on device
mf = cl.mem_flags
nx_cl = np.int32(nx)
a_dev = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_dev = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
c_dev = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c)

# run
lib.apb(queue, (nx,), None, nx_cl, a_dev, b_dev, c_dev)

# check
cl.enqueue_copy(queue, c, c_dev)
aa_equal(ref, c, 14)
