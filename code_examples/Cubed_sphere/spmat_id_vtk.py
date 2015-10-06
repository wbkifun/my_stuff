#------------------------------------------------------------------------------
# filename  : spmat_id_vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.10.6     Start
#
# description: 
#   Generate the VTK structured data format on the cubed-sphere
#   with weight of a sparse matrix
#   Implicit Diffusion
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from cube_vtk import CubeVTK2D



ne, ngq = 30, 4
print 'ne=%d, ngq=%d'%(ne, ngq)
cs_vtk = CubeVTK2D(ne, ngq)


uids = cs_vtk.cs_ncf.variables['uids']

ncf = nc.Dataset('./spmat_id_ne30ngq4.nc', 'r', format='NETCDF4')
spmat_size = len( ncf.dimensions['spmat_size'] )
dsts = ncf.variables['dsts'][:]
srcs = ncf.variables['srcs'][:]
wgts = ncf.variables['weights'][:]

#for dst, src, wgt in zip(dsts, srcs, wgts):
#    print dst, src, wgt


fw = np.ones(cs_vtk.up_size, 'f8')*(-10)    # background value

unique_dsts, index_dsts = np.unique(dsts, return_index=True)
dst_group = list(index_dsts) + [len(dsts)]


#--------------------------------------------------------
# Single target point
#--------------------------------------------------------
target_dst = 3
seq = np.where(unique_dsts==target_dst)[0]
start, end = dst_group[seq], dst_group[seq+1]
print 'src, uid, wgt, log10(|wgt|)'
s = 0
for i in xrange(start,end):
    src = srcs[i]
    wgt = wgts[i]
    uid = uids[src]
    s += wgt
    fw[uid] = np.log10(np.fabs(wgt))
    print '%d\t%d\t%g\t%g'%(src, uid, wgt, fw[uid])
print 'weight sum', s

variables = (('fw', 1, 1, fw.tolist()),)
#cs_vtk.write_with_variables('implicit_diffusion_weight_ne30ngq4.vtk', variables)


#--------------------------------------------------------
# Multiple target points
#--------------------------------------------------------
'''
for seq, dst in enumerate(unique_dsts[:100]):
    fw[:] = -10
    for i in xrange(dst_group[seq], dst_group[seq+1]):
        src = srcs[i]
        wgt = wgts[i]
        uid = uids[src]
        fw[uid] = np.log10(np.fabs(wgt))

    variables = (('fw', 1, 1, fw.tolist()),)
    cs_vtk.write_with_variables('./vtk/implicit_diffusion_weight_ne30ngq4_gid%.3d.vtk'%(dst), variables)
'''
