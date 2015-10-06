#------------------------------------------------------------------------------
# filename  : cube_partition_vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.10.6     Start
#
# description: 
#   Generate the VTK structured data format on the cubed-sphere
#   with domain decomposition using SFC
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from cube_vtk import CubeVTK2D
from cube_partition import CubePartition



ne, ngq = 30, 4
print 'ne=%d, ngq=%d'%(ne, ngq)

cs_vtk = CubeVTK2D(ne, ngq)


#---------------------------------------------------------------
# Read the grid indices
#---------------------------------------------------------------
fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)
cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')

ep_size = len( cs_ncf.dimensions['ep_size'] )
up_size = len( cs_ncf.dimensions['up_size'] )
gq_indices = cs_ncf.variables['gq_indices'][:]  # (ep_size,5)


for nproc in xrange(1,129):
    #---------------------------------------------------------------
    # Partitioning with SFC
    #---------------------------------------------------------------
    print 'nproc=%d'%(nproc)

    partition = CubePartition(ne, nproc)
    ep_ranks = np.zeros(ep_size, 'i4')
    cell_ranks = np.zeros(6*ne*ne*(ngq-1)*(ngq-1), 'i4')

    gq_eijs = gq_indices[:,:3] - 1
    idxs = gq_eijs[:,0]*ne*ne + gq_eijs[:,1]*ne + gq_eijs[:,2] 
    ep_ranks[:] = partition.elem_proc.ravel()[idxs]
    c_seq = 0
    for seq, (panel,ei,ej,gi,gj) in enumerate(gq_indices):
        if gi<ngq and gj<ngq:
            cell_ranks[c_seq] = ep_ranks[seq]
            c_seq += 1


    #---------------------------------------------------------------
    # Generate a VTK file
    #---------------------------------------------------------------
    # variables ('name', dimension, centering, arr)
    variables = (('ranks', 1, 0, cell_ranks.tolist()),)
    cs_vtk.write_with_variables('./vtk_cube_partition_ne%d/cs_partition_ne%d_nproc%d.vtk'%(ne,ne,nproc), variables)
