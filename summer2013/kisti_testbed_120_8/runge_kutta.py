from __future__ import division
import numpy

from sem_cores import add, rk4_add



class RungeKutta(object):
    def __init__(self, dt, inside):
        self.dt = dt
        self.tn = dt
        self.compute_rhs = inside.compute_rhs



    def allocate(self, shape, dtype):
        self.k1 = numpy.zeros(shape, dtype, order='F')
        self.k2 = numpy.zeros(shape, dtype, order='F')
        self.k3 = numpy.zeros(shape, dtype, order='F')
        self.k4 = numpy.zeros(shape, dtype, order='F')
        self.psi2 = numpy.zeros(shape, dtype, order='F')



    def update_rk4(self, psi):
        dt = self.dt
        tn = self.tn
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4
        psi2 = self.psi2

        '''
        compute_rhs = self.compute_rhs
        k1[:] = compute_rhs1(tn, psi) 
        k2[:] = compute_rhs(tn+0.5*dt, psi+0.5*dt*k1) 
        k3[:] = compute_rhs(tn+0.5*dt, psi+0.5*dt*k2) 
        k4[:] = compute_rhs(tn+dt, psi+dt*k3) 
        psi[:] = psi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        '''

        self.compute_rhs(psi, k1) 
  
        add(0.5*dt, k1, psi, psi2)
        self.compute_rhs(psi2, k2) 

        add(0.5*dt, k2, psi, psi2)
        self.compute_rhs(psi2, k3) 

        add(dt, k3, psi, psi2)
        self.compute_rhs(psi2, k4) 

        rk4_add(dt, k1, k2, k3, k4, psi)
        tn += dt



    def update_rk3(self, psi):
        dt = self.dt
        k1, k2 = self.k1, self.k2
        compute_rhs = self.compute_rhs

        k1[:] = psi + dt*compute_rhs(psi) 
        k2[:] = (3/4)*psi + (1/4)*k1 + (1/4)*dt*compute_rhs(k1) 
        psi[:] = (1/3)*psi + (2/3)*k2 + (2/3)*dt*compute_rhs(k2) 
