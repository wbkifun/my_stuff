#!/usr/bin/env python
from __future__ import division
import numpy
from numpy.random import rand

#from LJ_03 import force, solve, output_energy
from LJ_04 import force, solve, output_energy



n = 500         # Number of atoms, molecules
mt = 1000       # max time steps
dt = 10         # time interval (a.u.)
domain = 600    # domain size (a.u.)
ms = 0.00001    # max speed (a.u.)
em = 1822.88839*28.0134    # effective mass of N2
lje = 0.000313202          # Lennard-Jones epsilon of N2
ljs = 6.908841465          # Lennard-Jones sigma of N2


# allocate and initialize
x = domain * rand(n)
y = domain * rand(n)
z = domain * rand(n)
vx = ms * (rand(n) - 0.5)
vy = ms * (rand(n) - 0.5)
vz = ms * (rand(n) - 0.5)
fx = numpy.zeros(n)
fy = numpy.zeros(n)
fz = numpy.zeros(n)

energy = numpy.zeros(3)


# dynamics
for tstep in xrange(1,mt+1):
  force(lje, ljs, x, y, z, fx, fy, fz)
  solve(dt, em, x, y, z, vx, vy, vz, fx, fy, fz)

  #output_energy(tstep, em, lje, ljs, x, y, z, vx, vy, vz, energy)
  #print tstep, energy

output_energy(tstep, em, lje, ljs, x, y, z, vx, vy, vz, energy)
