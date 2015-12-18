#------------------------------------------------------------------------------
# filename  : cube_vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.18     Try with pyvtk, lack of documents
#             2015.9.21     Try with visit_writer, sample with 2D Gaussian
#             2015.12.8     Add make_spherical_harmonics
#
# description: 
#   Generate the VTK structured data format on the cubed-sphere
#   By using the visit_writer from VisIt
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import visit_writer

from util.grid.path import dir_cs_grid




class CubeVTK2D(object):
    def __init__(self, ne, ngq, rotated=False):
        self.ne = ne
        self.ngq = ngq
        self.rotated = rotated

        
        #-----------------------------------------------------
        # Read the grid and sparse matrix information
        #-----------------------------------------------------
        if rotated:
            cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d_rotated.nc'%(ne, ngq)
        else:
            cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)

        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')

        up_size = len( cs_ncf.dimensions['up_size'] )
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
        self.up_size = up_size

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



    def make_vtk_from_netcdf(self, target_fpath, ncf, varname_list):
        variables = list()
        for varname in varname_list:
            print '\t%s'%(varname)
            var = ncf.variables[varname][:]
            assert var.size == self.up_size, '%s size=%d is not same with up_size=%d'%(varname, var.size, self.up_size)

            variables.append( (varname, 1, 1, var.tolist()) )

        self.write_with_variables(target_fpath, variables)




def make_gaussian():
    '''
    A sample to test the CubeVTK2D
    '''
    ne, ngq = 30, 4
    cs_vtk = CubeVTK2D(ne, ngq)

    psi = np.zeros(cs_vtk.up_size, 'f8')
    lon0, lat0 = np.pi/2, -np.pi/5
    latlons = cs_vtk.cs_ncf.variables['latlons'][:]
    lats, lons = latlons[:,0], latlons[:,1]
    dist = np.arccos( np.sin(lat0)*np.sin(lats) \
            + np.cos(lat0)*np.cos(lats)*np.cos( np.fabs(lons-lon0) ) )    # r=1
    psi[:] = np.exp( -dist**2/(np.pi/50) )

    variables = (('psi', 1, 1, psi.tolist()),)
    cs_vtk.write_with_variables('gaussian_on_sphere.vtk', variables)




def make_spherical_harmonics(m, n, rotated):
    '''
    Generate a VTK file of the Spherical Harmonics using CubeVTK2D
    '''
    from scipy.special import sph_harm

    ne, ngq = 30, 4
    cs_vtk = CubeVTK2D(ne, ngq, rotated)

    psi = np.zeros(cs_vtk.up_size, 'f8')
    latlons = cs_vtk.cs_ncf.variables['latlons'][:]

    for i, (lat, lon) in enumerate(latlons):
        psi[i] = sph_harm(m, n, lon, np.pi/2-lat).real

    if cs_vtk.rotated:
        fname = 'spherical_harmonics_m%d_n%d_rotated.vtk'%(m,n)
    else:
        fname = 'spherical_harmonics_m%d_n%d.vtk'%(m,n)

    variables = (('psi', 1, 1, psi.tolist()),)
    cs_vtk.write_with_variables(fname, variables)




if __name__ == '__main__':
    #make_spherical_harmonics(2, 5, False)
    make_spherical_harmonics(2, 5, True)

    '''
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('nc_fpath', type=str, help='path of the NetCDF file')
    args = parser.parse_args()

    ncf = nc.Dataset(args.nc_fpath, 'r', format='NETCDF4')
    ne, ngq = ncf.ne, ncf.ngq
    cs_vtk = CubeVTK2D(ne, ngq)

    print 'Generate a VTK file from a NetCDF file'
    print 'ne=%d, ngq=%d'%(ne, ngq)
    print 'variables:'

    target_fpath = args.nc_fpath.replace('.nc', '.vtk')
    varname_list = [str(name) for name in ncf.variables.keys()]
    cs_vtk.make_vtk_from_netcdf(target_fpath, ncf, varname_list)
    '''
