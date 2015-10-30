#------------------------------------------------------------------------------
# filename  : dabp.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.30    start
#
# description: 
#   A simple OpenCL test
#   c = a + b
#------------------------------------------------------------------------------

from __future__ import division
import pyopencl as cl
import numpy as np
from numpy.testing import assert_array_equal as a_equal


src = ''' 
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void add(int nx, __global double *a, __global double *b, __global double *c) {
    int gid = get_global_id(0);

    if (gid >= nx) return;
    c[gid] = a[gid] + b[gid];
}
'''

platform_number = 0
device_number = 0


platforms = cl.get_platforms()
devices = platforms[platform_number].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context, devices[device_number])


nx = 1000000
a = np.random.rand(nx)
b = np.random.rand(nx)
c = np.zeros(nx)

mf = cl.mem_flags
nx_cl = np.int32(nx)
a_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_cl = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_cl = cl.Buffer(context, mf.WRITE_ONLY, c.nbytes)

lib = cl.Program(context, src).build()
event = lib.add(queue, (nx,), None, nx_cl, a_cl, b_cl, c_cl)
event.wait()
cl.enqueue_copy(queue, c, c_cl)
a_equal(a+b, c)
