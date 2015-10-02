#------------------------------------------------------------------------------
# filename  : runge_kutta.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.21     start
#
#
# description: 
#   Runge-Kutta method
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from multi_platform.machine_platform import MachinePlatform
from multi_platform.array_variable import Array, ArrayAs




class RungeKutta(object):
    def __init__(self, platform, size, dt):
        self.platform = platform
        self.size = size
        self.dt = dt


        #------------------------------------------------------------
        # Allocate
        #------------------------------------------------------------
        self.vtmp = vtmp = Array(platform, size, 'f8', 'vtmp')
        self.k1 = k1 = Array(platform, size, 'f8', 'k1')
        self.k2 = k2 = Array(platform, size, 'f8', 'k1')
        self.k3 = k3 = Array(platform, size, 'f8', 'k1')
        self.k4 = k4 = Array(platform, size, 'f8', 'k1')


        #------------------------------------------------------------
        # Prepare extern functions
        #------------------------------------------------------------
        src = open(__file__.replace('.py','.'+platform.code_type)).read()
        pyf = open(__file__.replace('.py','.pyf')).read()

        lib = platform.source_compile(src, pyf)
        self.daxpy = platform.get_function(lib, 'daxpy')
        self.rk4sum = platform.get_function(lib, 'rk4sum')

        self.daxpy.prepare('ioDOO', size, vtmp)     # add (t, k, var)
        self.rk4sum.prepare('idooooO', size, dt, k1, k2, k3, k4)    # add (var)



    def update_rk4(self, t, var, compute, communicate):
        dt = self.dt
        k1, k2, k3, k4, vtmp = self.k1, self.k2, self.k3, self.k4, self.vtmp

        # stage 1
        compute(t, var, k1)
        communicate(k1)

        # stage 2
        self.daxpy.prepared_call(0.5*dt, k1, var)
        compute(0.5*dt+t, vtmp, k2)
        communicate(k2)

        # stage 3
        self.daxpy.prepared_call(0.5*dt, k2, var)
        compute(0.5*dt+t, vtmp, k3)
        communicate(k3)

        # stage 4
        self.daxpy.prepared_call(dt, k3, var)
        compute(dt+t, vtmp, k4)
        communicate(k4)

        # update
        # var = var + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        self.rk4sum.prepared_call(var) 

        t += dt
