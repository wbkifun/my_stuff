#------------------------------------------------------------------------------
# filename  : dycore_sh.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.17    start
#
#
# description: 
#   Porting the KIM-SH dynamical core with MIP
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal

from cube_mpi import CubeGridMPI, CubeMPI
from cube_tensor import CubeTensor
from multi_platform.array_variable import Array, ArrayAs




class DycoreSH(object):
    def __init__(self, platform, cubegrid, cubetensor, nlev, **kwargs):
        self.platform = platform
        self.cubegrid = cubegrid
        self.cubetensor = cubetensor
        self.nlev = nlev

        ngq = cubegrid.ngq
        myrank = cubegrid.myrank
        nsize = cubegrid.local_ep_size


        #-------------------------------------------------------------
        # Constants
        #-------------------------------------------------------------
        r_earth = 6.376e6               # [m]

        if kwargs.has_key('dcmip21') or kwargs.has_key('dcmip22'):
            r_earth /= 500
        elif kwargs.has_key('dcmip31'):
            r_earth /= 125

        rr_earth = 1/r_earth


        #-------------------------------------------------------------
        # Vertical variables
        #-------------------------------------------------------------
        '''
        Hybrid level definitions: p = a*p0 + b*ps
        interfaces:   p(k) = hyai(k)*ps0 + hybi(k)*ps
        midpoints :   p(k) = hyam(k)*ps0 + hybm(k)*ps
        '''
        ps0 = 100000   # base state sfc pressure for level definitions
        hyai = Array(platform, nlev+1, 'f8', 'hyai', '', \
           'ps0 component of hybrid coordinate at interfaces')
        hybi = Array(platform, nlev+1, 'f8', 'hybi', '', \
           'ps component of hybrid coordinate at interfaces')
        hyam = Array(platform, nlev, 'f8', 'hyam', '', \
           'ps0 component of hybrid coordinate at midpoints')
        hybm = Array(platform, nlev, 'f8', 'hybm', '', \
           'ps0 component of hybrid coordinate at midpoints')
        etai = Array(platform, nlev+1, 'f8', 'etai', \
                'stored for conviencience')


        #-------------------------------------------------------------
        # Horizontal variables
        #-------------------------------------------------------------
        Dvv = ArrayAs(platform, cubetensor.Dvv)
        jac = ArrayAs(platform, cubetensor.jac)
        Dinv = ArrayAs(platform, cubetensor.Dinv)


        #-------------------------------------------------------------
        # Prognostic and Diagnostic variables
        #-------------------------------------------------------------
        ps_v = Array(platform, nsize, 'f8', 'ps_v', '', 'surface pressure on v grid')    
        v1 = Array(platform, nsize*nlev, 'f8', 'v1', 'm/s', 'longitude component of wind velocity')    
        v2 = Array(platform, nsize*nlev, 'f8', 'v2', 'm/s', 'latitude component of wind velocity')    
        grad_ps = Array(platform, nsize*2, 'f8', 'grad_ps')
        divdp = Array(platform, nsize*nlev, 'f8', 'divdp')
        vort = Array(platform, nsize*nlev, 'f8', 'vort')
        tmp3d = Array(platform, nsize*nlev, 'f8', 'tmp3d', 'for test')


        #-------------------------------------------------------------
        # Prepare extern functions
        #-------------------------------------------------------------
        src = open(__file__.replace('.py','.'+platform.code_type)).read()
        lib = platform.source_compile(src)
        self.compute_rhs = platform.get_function(lib, 'compute_rhs')

        #self.compute_rhs.prepare('iiiddoooOO', \
        #        nsize, nlev, ngq, rr_earth, ps0, \
        #        Dvv, jac, Dinv)
        #self.compute_rhs.prepare('iiiddooooooooOO', \
        #        nsize, nlev, ngq, rr_earth, ps0, \
        #        Dvv, jac, Dinv, hyai, hybi, hyam, hybm, etai)
        self.compute_rhs.prepare('iiiddooooooooOOOOOOO', \
                nsize, nlev, ngq, rr_earth, ps0, \
                Dvv, jac, Dinv, hyai, hybi, hyam, hybm, etai)


        #-------------------------------------------------------------
        # Public variables
        #-------------------------------------------------------------
        self.r_earth = r_earth
        self.rr_earth = rr_earth

        self.ps0 = ps0
        self.hyai = hyai
        self.hybi = hybi
        self.hyam = hyam
        self.hybm = hybm
        self.etai = etai

        self.Dvv = Dvv
        self.jac = jac
        self.Dinv = Dinv

        self.ps_v = ps_v
        self.grad_ps = grad_ps
        self.v1 = v1
        self.v2 = v2
        self.divdp = divdp
        self.vort = vort

        self.tmp3d = tmp3d



    def preset_variables(self, vname_list, vval_list):
        for vname, vval in zip(vname_list, vval_list):
            getattr(self, vname).set(vval)



    def run(self):
        #self.compute_rhs.prepared_call( \
        #        self.ps_v, self.grad_ps)
        self.compute_rhs.prepared_call( \
                self.ps_v, self.grad_ps, \
                self.v1, self.v2, \
                self.divdp, self.vort, \
                self.tmp3d)




if __name__ == '__main__':
    from multi_platform.device_platform import CPU_F90
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    ne, ngq = 30, 4
    nlev = 60


    platform = CPU_F90()
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank, homme_style=True)
    cubetensor = CubeTensor(cubegrid, homme_style=False)
    dycore = DycoreSH(platform, cubegrid, cubetensor, nlev)
