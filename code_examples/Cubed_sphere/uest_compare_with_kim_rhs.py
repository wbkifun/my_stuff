#------------------------------------------------------------------------------
# filename  : compare_with_kim.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.6     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import subprocess as subp

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup




ne, ngq = 30, 4



def compare_with_kim_rhs(myrank, nproc):
    '''
    Compare with KIM: CubeGridMPI, CubeTensor
    '''
    from cube_mpi import CubeGridMPI
    from cube_tensor import CubeTensor

    assert nproc==16


    #---------------------------------------------------------------------
    # Read reference variables from KIM
    #---------------------------------------------------------------------
    ref_ncf = nc.Dataset('compute_and_apply_rhs/rank_%d.nc'%(myrank), 'r')
    ref_nelemd = len( ref_ncf.dimensions['nelemd'] )
    ref_Dvv = ref_ncf.variables['Dvv'][:]
    ref_Dinv = ref_ncf.variables['Dinv'][:]
    ref_ps_v = ref_ncf.variables['ps_v'][:]
    ref_grad_ps = ref_ncf.variables['grad_ps'][:]


    #---------------------------------------------------------------------
    # Compare
    #---------------------------------------------------------------------
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank, homme_style=True)
    cubetensor = CubeTensor(cubegrid)

    # Note: Fortran array is column-major
    equal(ref_nelemd, cubegrid.local_nelem)
    aa_equal(ref_Dvv.ravel()*1e-1, cubetensor.dvvT*1e-1, 15)

    AI = cubetensor.AI.reshape(2,2,ngq,ngq,ref_nelemd)
    for ie in xrange(ref_nelemd):
        for gi in xrange(ngq):
            for gj in xrange(ngq):
                aa_equal(ref_Dinv[ie,gj,gi,:,:], AI[:,:,gj,gi,ie], 15)




def run_mpi(nproc):
    cmd = 'mpirun -np %d python test_compare_with_kim_rhs.py'%(nproc)
    proc = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = proc.communicate()
    #print stdout
    assert stderr == '', stderr




def test_compare_with_kim():
    nproc = 16
    func = with_setup(lambda:None, lambda:None)(lambda:run_mpi(nproc))
    func.description = 'Compare with KIM: compute_and_apply_rhs() (ne=%d, ngq=%d, nproc=%d)'%(ne, ngq, nproc)
    yield func




if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    compare_with_kim_rhs(myrank, nproc)
