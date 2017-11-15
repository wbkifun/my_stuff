import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import os
from numpy.testing import assert_array_equal as a_assert


# Show more compiler message
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


# setup
# -----
ctx = cl.create_some_context(interactive=True)
queue = cl.CommandQueue(ctx)

n = 15 * 10**6
a = cl.array.arange(queue, n, dtype=np.float64)

# create
# ------
knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")

# transform
# ---------
knl = lp.split_iname(knl, "i", 512, outer_tag="g.0", inner_tag="l.0")
#knl = lp.split_iname(knl, "i", 64, outer_tag="g.0", inner_tag="l.0")
knl = lp.set_options(knl, write_cl=False, write_wrapper=False)
print(knl)

typed_knl = lp.add_dtypes(knl, dict(a=np.float32))
code, _ = lp.generate_code(typed_knl)
print(code)


# execute
# -------
evt, (out,) = knl(queue, a=a)
a_assert(2*a.get(), out.get())
#print( np.linalg.norm(2*a.get()-out.get())==0 )

#cknl = lp.CompiledKernel(ctx, knl)
#print(cknl.get_highlighted_code({"a": np.float32}))
