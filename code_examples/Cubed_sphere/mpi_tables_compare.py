#------------------------------------------------------------------------------
# filename  : mpi_tables_compare.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.12    start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import argparse

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup
from mpi4py import MPI

comm = MPI.COMM_WORLD
myrank = comm.Get_rank()
nproc = comm.Get_size()



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('target', type=str, help='target directory')
args = parser.parse_args()

target = args.target + '/nproc%d_rank%d.nc'%(nproc,myrank)
ref = args.target + '.ref' + '/nproc%d_rank%d.nc'%(nproc,myrank)

try:
    r_ncf = nc.Dataset(ref   , 'r', format='NETCDF3_CLASSIC')
    t_ncf = nc.Dataset(target, 'r', format='NETCDF3_CLASSIC')
except Exception, e:
    print e
    print 'ref: %s'%ref
    print 'target: %s'%target
    import sys
    sys.exit()

for key in r_ncf.variables.keys():
    r_var = r_ncf.variables[key][:]
    t_var = t_ncf.variables[key][:]

    a_equal(r_var, t_var)
