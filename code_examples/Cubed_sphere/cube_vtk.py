#------------------------------------------------------------------------------
# filename  : cube_vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.18     Try with pyvtk, lack of documents
#             2013.9.21     Try with visit_writer, sample with 2D Gaussian
#
#
# description: 
#   Generate the VTK structured data format on the cubed-sphere
#   By using the visit_writer from VisIt
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import visit_writer


fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)




class CubeVTK2D(object):
    def __init__(self, ne, ngq):
        self.ne = ne
        self.ngq = ngq

        
        #-----------------------------------------------------
        # Read the grid and sparse matrix information
        #-----------------------------------------------------
        cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')

        uvp_size = len( cs_ncf.dimensions['uvp_size'] )
        gq_indices = cs_ncf.variables['gq_indices'][:]
        uids = cs_ncf.variables['uids'][:]
        xyzs = cs_ncf.variables['xyzs'][:]

        link_size = 6*ne*ne*(ngq-1)*(ngq-1)
        links = np.zeros((link_size,5), 'i4')


        #-----------------------------------------------------
        # Set the connectivity as vtk_cell_type is VTK_QUAD(=9)
        #-----------------------------------------------------
        ij2uid = dict()
        for seq, ij in enumerate(gq_indices):
            ij2uid[tuple(ij)] = uids[seq]

        link_seq = 0
        for seq, (p,ei,ej,gi,gj) in enumerate(gq_indices):
            if gi<ngq and gj<ngq:
                quad_ijs = [(p,ei,ej,gi,gj), \
                            (p,ei,ej,gi+1,gj), \
                            (p,ei,ej,gi+1,gj+1), \
                            (p,ei,ej,gi,gj+1)]
                links[link_seq][0] = visit_writer.quad
                links[link_seq][1:] = [ij2uid[ij] for ij in quad_ijs]
                link_seq += 1


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.cs_ncf = cs_ncf
        self.uvp_size = uvp_size

        self.pts = xyzs.ravel().tolist()
        self.links = links.tolist()



    def write_with_variables(self, fpath, variables, use_binary=True):
        '''
        variables : (('var1',dimension,centering,array),...)
        dimension : 1 for scalar or 3 for vector
        centering : 0 for cell-wise or 1 for point-wise
        '''
        visit_writer.WriteUnstructuredMesh(fpath, use_binary, \
                self.pts, self.links, variables)




if __name__ == '__main__':
    ne, ngq = 30, 4
    print 'ne=%d, ngq=%d'%(ne, ngq) 

    cs_vtk = CubeVTK2D(ne, ngq)

    # A sample with a 2D gaussian
    psi = np.zeros(cs_vtk.uvp_size, 'f8')
    lon0, lat0 = np.pi/2, -np.pi/5
    latlons = cs_vtk.cs_ncf.variables['latlons'][:]
    lats, lons = latlons[:,0], latlons[:,1]
    dist = np.arccos( np.sin(lat0)*np.sin(lats) \
            + np.cos(lat0)*np.cos(lats)*np.cos( np.fabs(lons-lon0) ) )    # r=1
    psi[:] = np.exp( -dist**2/(np.pi/50) )

    variables = (('psi', 1, 1, psi.tolist()),)
    cs_vtk.write_with_variables('gaussian_on_sphere.vtk', variables)
