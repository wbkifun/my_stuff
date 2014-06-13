#!/usr/bin/env python
from __future__ import division
import numpy
from numpy import int32 as i4, float64 as f8
import pyopencl as cl
from numpy.random import rand

#from LJ_03 import force, solve, output_energy
from LJ_04 import force, solve, output_energy


#------------------------------------------------------------------------------
# Platform, Device, Context and Queue
#------------------------------------------------------------------------------
platforms = cl.get_platforms()
intel = platforms[0]
devices = intel.get_devices()
context = cl.Context(devices)
cpu, mic = devices
queue_cpu = cl.CommandQueue(context, cpu)
queue_mic = cl.CommandQueue(context, mic)
 
device = cpu
queue = queue_cpu


#------------------------------------------------------------------------------
# setup
#------------------------------------------------------------------------------
n = 10000         # Number of atoms, molecules
mt = 1000       # max time steps
dt = 10         # time interval (a.u.)
domain = 600    # domain size (a.u.)
ms = 0.00001    # max speed (a.u.)
em = 1822.88839*28.0134    # effective mass of N2
lje = 0.000313202          # Lennard-Jones epsilon of N2
ljs = 6.908841465          # Lennard-Jones sigma of N2


'''
ls = local_size = 256
group_size = n//ls + 1
gs = global_size = group_size * local_size
print 'gs=%d, ls=%d, group_size=%d' % (gs, ls, group_size)
Gs, Ls = (gs,), (ls,)
'''
Gs, Ls = (n,), None


#------------------------------------------------------------------------------
# allocate and initialize
#------------------------------------------------------------------------------
x, y, z = [domain*rand(n) for i in xrange(3)]
vx, vy, vz = [ms*(rand(n)-0.5) for i in xrange(3)]
fx, fy, fz = [numpy.zeros(n) for i in xrange(3)]
ke_group = numpy.zeros(group_size)
pe_group = numpy.zeros(group_size)


# CL buffers
mf_rw = cl.mem_flags.READ_WRITE
x_buf, y_buf, z_buf = [cl.Buffer(context, mf_rw, f.nbytes) for f in [x,y,z]]
vx_buf, vy_buf, vz_buf = [cl.Buffer(context, mf_rw, f.nbytes) for f in [vx,vy,vz]]
fx_buf, fy_buf, fz_buf = [cl.Buffer(context, mf_rw, f.nbytes) for f in [fx,fy,fz]]
ke_group_buf = cl.Buffer(context, mf_rw, ke_group.nbytes)
pe_group_buf = cl.Buffer(context, mf_rw, pe_group.nbytes)

for t, t_buf in zip([x,y,z], [x_buf,y_buf,z_buf]): 
    cl.enqueue_copy(queue, t_buf, t)

for t, t_buf in zip([vx,vy,vz], [vx_buf,vy_buf,vz_buf]): 
    cl.enqueue_copy(queue, t_buf, t)


#------------------------------------------------------------------------------
# program and kernel
#------------------------------------------------------------------------------
kernels = open('LJ_10.cl').read()
prg = cl.Program(context, kernels).build(options='')

preferred_multiple = cl.Kernel(prg, 'force').get_work_group_info( \
	cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device) 

force_args = [i4(n), f8(lje), f8(ljs), x_buf, y_buf, z_buf, fx_buf, fy_buf, fz_buf]
solve_args = [i4(n), i4(dt), f8(em), x_buf, y_buf, z_buf, vx_buf, vy_buf, vz_buf, fx_buf, fy_buf, fz_buf]
energy_args = [i4(n), f8(em), f8(lje), f8(ljs), x_buf, y_buf, z_buf, vx_buf, vy_buf, vz_buf, ke_group_buf, pe_group_buf]


#------------------------------------------------------------------------------
# dynamics
#------------------------------------------------------------------------------
for tstep in xrange(1,mt+1):
    prg.force(queue, Gs, Ls, *force_args)
    prg.solve(queue, Gs, Ls, *solve_args)

    #prg.energy(queue, Gs, Ls, *energy_args)
    #cl.enqueue_copy(queue, ke_group, ke_group_buf)
    #cl.enqueue_copy(queue, pe_group, pe_group_buf)
	#ke, pe = ke_group.sum(), pe_group.sum()
	#te = ke + pe
	#print tstep, ke, pe, te 


prg.energy(queue, Gs, Ls, *energy_args)
cl.enqueue_copy(queue, ke_group, ke_group_buf)
cl.enqueue_copy(queue, ke_group, pe_group_buf)
ke, pe = ke_group.sum(), pe_group.sum()
te = ke + pe
print tstep, ke, pe, te 
