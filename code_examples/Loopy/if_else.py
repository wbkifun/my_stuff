import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
#import pyopencl.clrandom
import os
from numpy.testing import assert_array_equal as a_assert


# Show more compiler message
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'


#
# OpenCL Environment
#
#ctx = cl.create_some_context(interactive=True)
#queue = cl.CommandQueue(ctx)

vendor_name, device_type = 'Intel', 'CPU'
#vendor_name, device_type = 'NVIDIA', 'GPU'

platforms = cl.get_platforms()
platform = [p for p in platforms if vendor_name in p.vendor][0]
devices = platform.get_devices()
device = [d for d in devices if cl.device_type.to_string(d.type)==device_type][0]
print(platform)
print(device)
ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx, device)


#
# Setup
#
nx = 10
dtype = np.float32
#a = cl.clrandom.rand(queue, nx, dtype)
a = cl.array.arange(queue, nx, dtype=dtype)
b = cl.array.zeros(queue, nx, dtype)

#
# Kernel
#
knl = lp.make_kernel(
        "{ [i]: 0<=i<nx }",
        """
        double(s) := 2*s

        b[i] = if(i<nx/2, double(a[i]), a[i])
        """
        ) 

# transform
# ---------
wgs = 512   # Work Group Size
knl = lp.fix_parameters(knl, nx=nx)
knl = lp.set_options(knl, write_cl=True, write_wrapper=False)
print(knl)

#typed_knl = lp.add_dtypes(knl, dict(a=np.float32))
#code, _ = lp.generate_code(typed_knl)
#print(code)


# execute
# -------
evt, (out,) = knl(queue, a=a, b=b)
#a_assert(2*a.get(), b.get())
#print( np.linalg.norm(2*a.get()-out.get())==0 )
print(a.get())
print(b.get())

#cknl = lp.CompiledKernel(ctx, knl)
#print(cknl.get_highlighted_code({"a": np.float32}))
