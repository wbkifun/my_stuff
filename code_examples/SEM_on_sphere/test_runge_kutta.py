#------------------------------------------------------------------------------
# filename  : test_rk4.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.21     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_


from multi_platform.machine_platform import MachinePlatform
from multi_platform.array_variable import Array, ArrayAs




class PreSetup(object):
    def __init__(self):
        self.nx = nx = 100
        self.tmax = tmax = 1000
        self.dt = dt = 1/tmax
        self.yinit = y0 = np.random.rand(nx)

        self.func = lambda t,y: -(t+1)*y
        self.exact_func = lambda t: y0*np.exp(-0.5*t**2 - t)




def test_rk4_exact():
    '''
    RK4: Exact solution dy/dt=-(t+1)*y => y(t)=y(0)*exp(-0.5*t-1)*t
    '''

    ps = PreSetup()
    nx, dt, tmax, func = ps.nx, ps.dt, ps.tmax, ps.func

    y = np.zeros(nx)
    k1 = np.zeros_like(y)
    k2 = np.zeros_like(y)
    k3 = np.zeros_like(y)
    k4 = np.zeros_like(y)

    #----------------------------------------------------
    # RK4
    #----------------------------------------------------
    t = 0
    y[:] = ps.yinit
    for tstep in xrange(tmax):
        k1[:] = func(t, y)
        k2[:] = func(t+0.5*dt, y+0.5*dt*k1) 
        k3[:] = func(t+0.5*dt, y+0.5*dt*k2) 
        k4[:] = func(t+dt, y+dt*k3) 

        y += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        t += dt

    aa_equal(ps.exact_func(t), y, 13)




#============================================================================
exact_func_pyf = '''
python module $MODNAME
  interface
    subroutine func(n, t, y, ret)
      intent(c) :: func
      intent(c)
      integer, required, intent(in) :: n
      real(8), intent(in) :: t, y(n)
      real(8), intent(inout) :: ret(n)
    end subroutine
  end interface
end python module
    '''



exact_func_f90 = '''
SUBROUTINE func(n, t, y, ret)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: t, y(n)
  REAL(8), INTENT(INOUT) :: ret(n)

  INTEGER :: i

  DO i=1,n
    ret(i) = -(t + 1)*y(i)
  END DO
END SUBROUTINE
    '''



exact_func_c = '''
void func(int n, double t, double *y, double *ret) {
    int i;
    for(i=0; i<n; i++) ret[i] = -(t + 1)*y[i];
}
    '''



exact_func_cl = '''
__kernel void func(int n, double t, __global double *y, __global double *ret) {
    int i = get_global_id(0);

    if (i >= n) return;
    ret[i] = -(t + 1)*y[i];
}
    '''



exact_func_cu = '''
__global__ void func(int n, double t, double *y, double *ret) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;
    ret[i] = -(t + 1)*y[i];
}
    '''




def check_rk4_exact_multi(platform, func_src, func_pyf, daxpy_src, daxpy_pyf):
    #----------------------------------------------------
    # Allocate
    #----------------------------------------------------
    ps = PreSetup()
    nx, dt, tmax, yinit = ps.nx, ps.dt, ps.tmax, ps.yinit

    y = ArrayAs(platform, yinit, 'y')
    ytmp = Array(platform, nx, 'f8', 'ytmp')

    k1 = Array(platform, nx, 'f8', 'k1')
    k2 = Array(platform, nx, 'f8', 'k2')
    k3 = Array(platform, nx, 'f8', 'k3')
    k4 = Array(platform, nx, 'f8', 'k4')


    #----------------------------------------------------
    # Core function
    #----------------------------------------------------
    lib = platform.source_compile(func_src, func_pyf)
    func = platform.get_function(lib, 'func')
    func.prepare('iDOO', nx)   # (t, y, ret)


    #----------------------------------------------------
    # DAXPY(double a*x+y) function
    #----------------------------------------------------
    lib = platform.source_compile(daxpy_src, daxpy_pyf)
    daxpy = platform.get_function(lib, 'daxpy')
    rk4sum = platform.get_function(lib, 'rk4sum')
    daxpy.prepare('iooDO', nx, y, ytmp)
    rk4sum.prepare('idooooo', nx, dt, k1, k2, k3, k4, y)


    #----------------------------------------------------
    # RK4
    #----------------------------------------------------
    t = 0
    for tstep in xrange(tmax):
        func.prepared_call(t, y, k1)

        daxpy.prepared_call(0.5*dt, k1)
        func.prepared_call(t+0.5*dt, ytmp, k2) 

        daxpy.prepared_call(0.5*dt, k2)
        func.prepared_call(t+0.5*dt, ytmp, k3) 

        daxpy.prepared_call(dt, k3)
        func.prepared_call(t+dt, ytmp, k4) 

        # y += (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        rk4sum.prepared_call()
        t += dt

    aa_equal(ps.exact_func(t), y.get(), 13)





def test_rk4_exact_cpu_f90():
    '''
    RK4: Exact solution dy/dt=-(t+1)*y on CPU-Fortran90
    '''
    daxpy_src = '''
SUBROUTINE daxpy(n, y, ret, a, x)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a, x(n), y(n)
  REAL(8), INTENT(INOUT) :: ret(n)

  INTEGER :: i

  DO i=1,n
    ret(i) = a*x(i) + y(i)
  END DO
END SUBROUTINE


SUBROUTINE rk4sum(n, dt, k1, k2, k3, k4, ret)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: dt, k1(n), k2(n), k3(n), k4(n)
  REAL(8), INTENT(INOUT) :: ret(n)

  INTEGER :: i

  DO i=1,n
    ret(i) = ret(i) + (dt/6)*(k1(i) + 2*k2(i) + 2*k3(i) + k4(i))
  END DO
END SUBROUTINE
    '''

    daxpy_pyf = '''
python module $MODNAME
  interface
    subroutine daxpy(n, y, ret, a, x)
      integer, required, intent(in) :: n
      real(8), intent(in) :: a, x(n), y(n)
      real(8), intent(inout) :: ret(n)
    end subroutine

    subroutine rk4sum(n, dt, k1, k2, k3, k4, ret)
      integer, required, intent(in) :: n
      real(8), intent(in) :: dt, k1(n), k2(n), k3(n), k4(n)
      real(8), intent(inout) :: ret(n)
    end subroutine
  end interface
end python module
    '''

    platform = MachinePlatform('CPU', 'f90')
    check_rk4_exact_multi(platform, exact_func_f90, exact_func_pyf, daxpy_src, daxpy_pyf)




def test_rk4_exact_cpu_c():
    '''
    RK4: Exact solution dy/dt=-(t+1)*y on CPU-C
    '''
    daxpy_src = '''
void daxpy(int n, double *y, double *ret, double a, double *x) {
    int i;
    for(i=0; i<n; i++) ret[i] = a*x[i] + y[i];
}


void rk4sum(int n, double dt, double *k1, double *k2, double *k3, double *k4, double *ret) {
    int i;
    for(i=0; i<n; i++) 
        ret[i] += (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
}
    '''

    daxpy_pyf = '''
python module $MODNAME
  interface
    subroutine daxpy(n, y, ret, a, x)
      intent(c) :: daxpy
      intent(c)
      integer, required, intent(in) :: n
      real(8), intent(in) :: a, x(n), y(n)
      real(8), intent(inout) :: ret(n)
    end subroutine

    subroutine rk4sum(n, dt, k1, k2, k3, k4, ret)
      intent(c) :: rk4sum
      intent(c)
      integer, required, intent(in) :: n
      real(8), intent(in) :: dt, k1(n), k2(n), k3(n), k4(n)
      real(8), intent(inout) :: ret(n)
    end subroutine
  end interface
end python module
    '''

    platform = MachinePlatform('CPU', 'c')
    check_rk4_exact_multi(platform, exact_func_c, exact_func_pyf, daxpy_src, daxpy_pyf)




def test_rk4_exact_cpu_cl():
    '''
    RK4: Exact solution dy/dt=-(t+1)*y on CPU-OpenCL
    '''
    daxpy_src = '''
__kernel void daxpy(int n, __global double *y, __global double *ret, double a, __global double *x) {
    int i = get_global_id(0);

    if (i >= n) return;
    ret[i] = a*x[i] + y[i];
}


__kernel void rk4sum(int n, double dt, __global double *k1, __global double *k2, __global double *k3, __global double *k4, __global double *ret) {
    int i = get_global_id(0);

    if (i >= n) return;
    ret[i] += (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
}
    '''

    platform = MachinePlatform('CPU', 'cl')
    check_rk4_exact_multi(platform, exact_func_cl, '', daxpy_src, '')




def test_rk4_exact_gpu_cu():
    '''
    RK4: Exact solution dy/dt=-(t+1)*y on NVIDIA GPU-CUDA
    '''
    daxpy_src = '''
__global__ void daxpy(int n, double *y, double *ret, double a, double *x) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;
    ret[i] = a*x[i] + y[i];
}


__global__ void rk4sum(int n, double dt, double *k1, double *k2, double *k3, double *k4, double *ret) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) return;
    ret[i] += (dt/6)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
}
    '''

    platform = MachinePlatform('NVIDIA GPU', 'cu')
    check_rk4_exact_multi(platform, exact_func_cu, '', daxpy_src, '')





def check_RungeKutta4_exact_multi(platform, func_src, func_pyf):
    from runge_kutta import RungeKutta


    #----------------------------------------------------
    # Allocate
    #----------------------------------------------------
    ps = PreSetup()
    nx, dt, tmax, yinit = ps.nx, ps.dt, ps.tmax, ps.yinit

    y = ArrayAs(platform, yinit, 'y')

    rk = RungeKutta(platform, nx, dt)


    #----------------------------------------------------
    # Core function
    #----------------------------------------------------
    lib = platform.source_compile(func_src, func_pyf)
    func_core = platform.get_function(lib, 'func')
    func_core.prepare('iDOO', nx)   # (t, y, ret)

    func = lambda t, y, ret: func_core.prepared_call(t,y,ret)
    comm = lambda k: None


    #----------------------------------------------------
    # RK4
    #----------------------------------------------------
    t = 0
    for tstep in xrange(tmax):
        rk.update_rk4(t, y, func, comm)
        t += dt

    aa_equal(ps.exact_func(t), y.get(), 13)




def test_RungeKutta4_exact_cpu_f90():
    '''
    RungeKutta4: Exact solution dy/dt=-(t+1)*y on CPU-Fortran90
    '''
    platform = MachinePlatform('CPU', 'f90')
    check_RungeKutta4_exact_multi(platform, exact_func_f90, exact_func_pyf)




def test_RungeKutta4_exact_cpu_c():
    '''
    RungeKutta4: Exact solution dy/dt=-(t+1)*y on CPU-C
    '''
    platform = MachinePlatform('CPU', 'c')
    check_RungeKutta4_exact_multi(platform, exact_func_c, exact_func_pyf)




def test_RungeKutta4_exact_cpu_cl():
    '''
    RungeKutta4: Exact solution dy/dt=-(t+1)*y on CPU-OpenCL
    '''
    platform = MachinePlatform('CPU', 'cl')
    check_RungeKutta4_exact_multi(platform, exact_func_cl, '')




def test_RungeKutta4_exact_gpu_cu():
    '''
    RungeKutta4: Exact solution dy/dt=-(t+1)*y on NVIDIA GPU-CUDA
    '''
    platform = MachinePlatform('NVIDIA GPU', 'cu')
    check_RungeKutta4_exact_multi(platform, exact_func_cu, '')
