from __future__ import division
import numpy
import sys
from numpy import pi, sin, cos

from pykgm.trial.parallel_sfc.io_preprocess import load_preprocess_netcdf
from pykgm.trial.parallel_sfc.cube_partition_run import CubePartitionRun

from pykgm.share.physical_constant import rotating_velocity as omega
from pykgm.share.physical_constant import earth_radius as rearth
from pykgm.share.physical_constant import gravitational_const as gconst

from shallow_water import State, InsideElems, RungeKutta




#----------------------------------------------
# setup
#----------------------------------------------
N = 4          # elements / axis
ngll = 4        # GLL points / axis / element
#cfl = 0.2       # Courant-Friedrichs-Lewy condition 
dt = 800
nproc = 1
rank = 1

# load the preprocessed netcdf file
ncf = load_preprocess_netcdf(N, ngll, nproc, rank)
nelem = ncf.nelem    # total elements
min_dx = ncf.min_dx*rearth
lonlat_coord = ncf.variables['lonlat_coord'][:]
lats = lonlat_coord[1,:,:,:]

# state variables
state = State(ncf)



#----------------------------------------------
# initialize the states
#----------------------------------------------
u0 = 2*pi*rearth/(12*24*3600)  # 12 days
h0 = 2.94*1e4/gconst

state.fcor[:,:,:] = 2*omega*sin(lats)
state.u[:,:,0,:] = u0*cos(lats)
state.h[:,:,0,:] = h0 - (2*rearth*omega + u0)*u0*sin(lats)**2 / (2*gconst)



#----------------------------------------------
# spectral element method
#----------------------------------------------
interact = CubePartitionRun(N, ngll, ncf, mpi_comm=None)
inside = InsideElems(ncf, state, interact)

# minimum dx, dt
#max_v = u0
#dt = cfl*min_dx/max_v
tint = RungeKutta(dt, inside.compute_rhs)
tint.allocate(state.shape)



#----------------------------------------------
# print the setup information
#----------------------------------------------
print '-'*47
print 'N\t\t', N
print 'ngll\t\t', ngll
print 'nelem\t\t', nelem
print 'u0\t\t', u0
print 'h0\t\t', h0
#print 'cfl\t\t', cfl
print 'min_dx\t\t', min_dx
print 'dt\t\t', dt

tmax = int( numpy.ceil(12*24*3600/dt) )
#tmax = 1000
tgap = 100

print 'tmax\t\t', tmax
print 'tgap\t\t', tgap
print '-'*47
print ''

numpy.save('./run/%.6d_rank%d_h.npy' % (0, rank), state.h)



#----------------------------------------------
# time loop
#----------------------------------------------
for tstep in xrange(1,tmax+1):
    tint.update_rk4(state.u, state.v, state.h)

    if tstep%tgap == 0:
        print 'tstep=\t%d/%d (%g %s)\r' % (tstep, tmax, tstep/tmax*100, '%'),
        sys.stdout.flush()

        #numpy.save('./run/%.6d_h.npy' % (tstep, state.h))
        if tstep == 1:
            f = open('./run/%.6d.ascii'%tstep, 'w')
        elif tstep == tmax:
            f.close()
        else:
            state.save_ascii(tstep, f)

print ''
