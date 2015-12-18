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
import sys

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup


ne, ngq = 30, 4
nlev = 60




def compare_with_kim_rhs(myrank, nproc):
    '''
    Compare with KIM: CubeGridMPI, CubeTensor
    '''
    from cube_mpi import CubeGridMPI
    from cube_tensor import CubeTensor
    from dycore_sh import DycoreSH
    from multi_platform.device_platform import CPU_F90


    assert nproc==16


    platform = CPU_F90()
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank, homme_style=True)
    cubetensor = CubeTensor(cubegrid, homme_style=True)
    dycore = DycoreSH(platform, cubegrid, cubetensor, nlev, dcmip21='on')


    #---------------------------------------------------------------------
    # Read reference variables from KIM
    #---------------------------------------------------------------------
    #base_dir = './kim_inspect/compute_and_apply_rhs'
    base_dir = '/nas2/user/khkim/kim_inspect/compute_and_apply_rhs'
    kim_ncf = nc.Dataset(base_dir+'/rank%d.nc'%(myrank), 'r')

    kim_nelemd = len( kim_ncf.dimensions['nelemd'] )
    kim_Dvv = kim_ncf.variables['Dvv'][:]           # (ngq,ngq)
    kim_Dinv = kim_ncf.variables['Dinv'][:]         # (nelem,ngq,ngq,2,2)
    kim_metdet = kim_ncf.variables['metdet'][:]     # (nelem,ngq,ngq)
    kim_hyai = kim_ncf.variables['hvcoord_hyai'][:] # (nlev+1)
    kim_hybi = kim_ncf.variables['hvcoord_hybi'][:] # (nlev+1)
    kim_hyam = kim_ncf.variables['hvcoord_hyam'][:] # (nlev)
    kim_hybm = kim_ncf.variables['hvcoord_hybm'][:] # (nlev)
    kim_etai = kim_ncf.variables['hvcoord_etai'][:] # (nlev+1)

    kim_ps_v = kim_ncf.variables['ps_v'][:]         # (nelem,ngq,ngq)
    kim_grad_ps = kim_ncf.variables['grad_ps'][:]   # (nelem,ngq,ngq,2)
    kim_v1 = kim_ncf.variables['v1'][:]             # (nelem,nlev,ngq,ngq)
    kim_v2 = kim_ncf.variables['v2'][:]             # (nelem,nlev,ngq,ngq)
    kim_divdp = kim_ncf.variables['divdp'][:]       # (nelem,nlev,ngq,ngq)
    kim_vort = kim_ncf.variables['vort'][:]         # (nelem,nlev,ngq,ngq)
    kim_tmp3d = kim_ncf.variables['tmp3d'][:]       # (nelem,nlev,ngq,ngq)

    ref_nelemd = kim_nelemd
    ref_Dvv = kim_Dvv.reshape(-1)
    ref_Dinv = kim_Dinv.reshape(-1)
    ref_metdet = kim_metdet.reshape(-1)
    ref_hyai = kim_hyai
    ref_hybi = kim_hybi
    ref_hyam = kim_hyam
    ref_hybm = kim_hybm
    ref_etai = kim_etai

    ref_ps_v = kim_ps_v.reshape(-1)
    ref_grad_ps = kim_grad_ps.reshape(-1)
    ref_v1 = kim_v1.reshape(-1)
    ref_v2 = kim_v2.reshape(-1)
    ref_divdp = kim_divdp.reshape(-1)
    ref_vort = kim_vort.reshape(-1)
    ref_tmp3d = kim_tmp3d.reshape(-1)


    #---------------------------------------------------------------------
    # Compare with KIM
    # Note: Fortran array is column-major
    #---------------------------------------------------------------------
    #
    # local_nelem
    #
    equal(ref_nelemd, cubegrid.local_nelem)


    #
    # dvv
    #
    aa_equal(ref_Dvv, cubetensor.Dvv, 15)


    #
    # Dinv
    #
    '''
    if myrank==0:
        print ''
        print kim_Dinv[0,0,0,0,0], cubetensor.Dinv[0]
        print kim_Dinv[0,0,0,0,1], cubetensor.Dinv[1]
        print kim_Dinv[0,0,0,1,0], cubetensor.Dinv[2]
        print kim_Dinv[0,0,0,1,1], cubetensor.Dinv[3]
    '''
    aa_equal(ref_Dinv, cubetensor.Dinv, 9)      # homme stype: 13


    #
    # metdet (jac)
    #
    aa_equal(ref_metdet, cubetensor.jac, 15)

    #
    # gradient
    #
    #dycore.preset_variables(['Dvv', 'Dinv', 'ps_v'], [ref_Dvv, ref_Dinv, ref_ps_v])
    dycore.preset_variables(['Dvv', 'Dinv', 'jac', 'ps_v', 'v1', 'v2', 'hyai', 'hybi', 'hyam', 'hybm', 'etai'], [ref_Dvv, ref_Dinv, ref_metdet, ref_ps_v, ref_v1, ref_v2, ref_hyai, ref_hybi, ref_hyam, ref_hybm, ref_etai])
    dycore.run()

    #if myrank == 0:
    #    aa_equal(ref_grad_ps[::2], dycore.grad_ps.get()[::2], 12)
    #    aa_equal(ref_grad_ps[1::2], dycore.grad_ps.get()[1::2], 12)
    a_equal(ref_grad_ps, dycore.grad_ps.get())
    a_equal(ref_tmp3d, dycore.tmp3d.get())
    a_equal(ref_divdp, dycore.divdp.get())
    #a_equal(ref_vort, dycore.vort.get())




def run_mpi(nproc):
    cmd = 'mpirun -np %d python test_compare_with_kim_rhs.py'%(nproc)
    proc = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = proc.communicate()
    print stdout
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
