from __future__ import division
import numpy
import atexit
from numpy.testing import assert_array_equal as assert_ae

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from source_module_f90 import get_module_f90



#------------------------------------------------------------------------------
# initialize CUDA
#------------------------------------------------------------------------------
cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()
atexit.register(ctx.pop)



#------------------------------------------------------------------------------
# Fortran source
#------------------------------------------------------------------------------
src_f90 = '''
SUBROUTINE add(nx, a, b, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  DOUBLE PRECISION, DIMENSION(nx), INTENT(IN) :: a, b
  DOUBLE PRECISION, DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: i

  DO i=1,nx
    c(i) = a(i) + b(i)
  END DO
END SUBROUTINE


SUBROUTINE multiply(nx, a, b, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  DOUBLE PRECISION, DIMENSION(nx), INTENT(IN) :: a, b
  DOUBLE PRECISION, DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: i

  DO i=1,nx
    c(i) = a(i)*b(i)
  END DO
END SUBROUTINE


SUBROUTINE multiply_add(nx, a, b, c)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: nx
  DOUBLE PRECISION, DIMENSION(nx), INTENT(IN) :: a, b
  DOUBLE PRECISION, DIMENSION(nx), INTENT(INOUT) :: c

  INTEGER :: i

  DO i=1,nx
    c(i) = c(i) + a(i)*b(i)
  END DO
END SUBROUTINE
'''

mod = get_module_f90(src_f90)
add_f90 = mod.add
multiply_f90 = mod.multiply
multiply_add_f90 = mod.multiply_add



#------------------------------------------------------------------------------
# CUDA source
#------------------------------------------------------------------------------
cu_src = '''
__global__ void add(int nx, double *a, double *b, double *c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nx) {
        c[tid] = a[tid] + b[tid];
    }
}


__global__ void multiply(int nx, double *a, double *b, double *c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nx) {
        c[tid] = a[tid] * b[tid];
    }
}


__global__ void multiply_add(int nx, double *a, double *b, double *c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < nx) {
        c[tid] += a[tid] * b[tid];
    }
}
'''

#mod = SourceModule(cu_src)
mod = SourceModule(cu_src, options=['--fmad=false'])
add_gpu = mod.get_function('add')
multiply_gpu = mod.get_function('multiply')
multiply_add_gpu = mod.get_function('multiply_add')



#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
nx = 100000
a = numpy.random.rand(nx)
b = numpy.random.rand(nx)
c = numpy.random.rand(nx)


a_f90 = a.copy()
b_f90 = b.copy()
c_f90 = c.copy()


a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.to_device(c)
c_from_gpu = numpy.empty(nx)

block = (256,1,1)
grid = (nx//256+1,1)



#------------------------------------------------------------------------------
# Compare
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Multiply
print 'Multiply'
c[:] = a*b

#assert_ae(c, c_f90)
multiply_f90(a_f90, b_f90, c_f90)
assert_ae(c, c_f90)

#assert_ae(c, c_from_gpu)
multiply_gpu(numpy.int32(nx), a_gpu, b_gpu, c_gpu, block=block, grid=grid)
cuda.memcpy_dtoh(c_from_gpu, c_gpu)
assert_ae(c, c_from_gpu)


#------------------------------------------------------------------------------
# Add
print 'Add'
c[:] = a + b

#assert_ae(c, c_f90)
add_f90(a_f90, b_f90, c_f90)
assert_ae(c, c_f90)

#assert_ae(c, c_from_gpu)
add_gpu(numpy.int32(nx), a_gpu, b_gpu, c_gpu, block=block, grid=grid)
cuda.memcpy_dtoh(c_from_gpu, c_gpu)
assert_ae(c, c_from_gpu)


#------------------------------------------------------------------------------
# Multiply-Add
print 'Multiply-Add'
c[:] = c + a*b

#assert_ae(c, c_f90)
multiply_add_f90(a_f90, b_f90, c_f90)
assert_ae(c, c_f90)

#assert_ae(c, c_from_gpu)
multiply_add_gpu(numpy.int32(nx), a_gpu, b_gpu, c_gpu, block=block, grid=grid)
cuda.memcpy_dtoh(c_from_gpu, c_gpu)
assert_ae(c, c_from_gpu)
