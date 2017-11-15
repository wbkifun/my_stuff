import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
import os
from numpy.testing import assert_array_equal as a_assert


# Show more compiler message
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'


#
# OpenCL Environment
#
#ctx = cl.create_some_context(interactive=True)
#queue = cl.CommandQueue(ctx)

#vendor_name, device_type = 'Intel', 'CPU'
vendor_name, device_type = 'NVIDIA', 'GPU'

platforms = cl.get_platforms()
platform = [p for p in platforms if vendor_name in p.vendor][0]
devices = platform.get_devices()
device = [d for d in devices if cl.device_type.to_string(d.type)==device_type][0]
ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx, device)


#
# Setup
#
ne, ngq, nlev = 30, 4, 50
nelem = ne*ne*6
dtype = np.float64
a = cl.clrandom.rand(queue, (nelem,nlev+1,ngq*ngq), dtype)
b = cl.array.zeros(queue, (nelem,nlev+1,ngq*ngq), dtype)

#
# Kernel
#
knl = lp.make_kernel(
        "{ [ie,k,ji]: 0<=ie<nelem and 0<=k<=nlev and 0<=ji<16 }",
        """
        b[ie,k,ji] = 2*a[ie,k,ji] {if=k>0}
        """)

# transform
# ---------
#knl = lp.split_iname(knl, "i", 512, outer_tag="g.0", inner_tag="l.0")
#knl = lp.split_iname(knl, "i", 64, outer_tag="g.0", inner_tag="l.0")
#knl = lp.join_inames(knl, ['j','i'], 'ji', tag='unr')

wgs = 512   # Work Group Size
knl = lp.fix_parameters(knl, nelem=nelem, nlev=nlev)
#knl = lp.split_iname(knl, 'k', wgs//(ngq*ngq))
#knl = lp.tag_inames(knl, dict(k_inner='l.1', ji='l.0'))
#knl = lp.tag_inames(knl, dict(ji='unr'))
#knl = lp.set_loop_priority(knl, 'ie,k_outer,k_inner,ji')
knl = lp.set_loop_priority(knl, 'ie,k,ji')
knl = lp.set_options(knl, write_cl=True, write_wrapper=False)
print(knl)

#typed_knl = lp.add_dtypes(knl, dict(a=np.float32))
#code, _ = lp.generate_code(typed_knl)
#print(code)


# execute
# -------
evt, (out,) = knl(queue, a=a, b=b)
a_assert(2*a.get(), b.get())
#print( np.linalg.norm(2*a.get()-out.get())==0 )

#cknl = lp.CompiledKernel(ctx, knl)
#print(cknl.get_highlighted_code({"a": np.float32}))
