#------------------------------------------------------------------------------
# filename  : cube_vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.8     start
#
# description: 
#   Generate the VTK regular or retilinear data format from the latlon
#   By using the visit_writer from VisIt
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import visit_writer




class LatlonVTK2D(object):
    def __init__(self, nlat, nlon, ll_type='regular', geo='sphere'):
        '''
        Note: The latlon grid should not include the pole.
        Support ll_type: regular, gaussian
        Support shape: sphere, plane
        '''
        self.nlat = nlat
        self.nlon = nlon
        self.ll_type = ll_type
        self.geo = geo  

        self.nsize = nlat*nlon


        #-----------------------------------------------------
        # Generate latlon grid
        #-----------------------------------------------------
        lons = np.linspace(0, 2*np.pi, nlon+1)[:-1]

        if ll_type == 'regular':
            lats = np.linspace(-np.pi/2, np.pi/2, nlat+2)[1:-1]

        elif ll_type == 'gaussian':
            import spharm   # NCAR SPHEREPACK
            degs, wts = spharm.gaussian_lats_wts(nlat)
            lats = np.deg2rad(degs[::-1])    # convert to south pole first

        else:
            raise ValueError, 'Wrong ll_type=%s. Support ll_type: regular, gaussian'%(ll_type)

        self.lons = lons
        self.lats = lats


        #-----------------------------------------------------
        # Set the connectivity
        #-----------------------------------------------------
        if geo == 'sphere':
            # vtk_cell_type is VTK_QUAD(=9)

            xyzs = np.zeros((nlat*nlon,3), 'f8')
            seq = 0
            for lat in lats:
                for lon in lons:
                    xyzs[seq,0] = np.cos(lat)*np.cos(lon)
                    xyzs[seq,1] = np.cos(lat)*np.sin(lon)
                    xyzs[seq,2] = np.sin(lat)
                    seq += 1

            link_size = (nlat-1)*nlon
            links = np.zeros((link_size,5), 'i4')

            link_seq = 0
            for j in xrange(nlat-1):
                for i in xrange(nlon):
                    ip = i+1 if i<nlon-1 else 0
                    seq1 = j*nlon + i
                    seq2 = j*nlon + ip
                    seq3 = (j+1)*nlon + ip
                    seq4 = (j+1)*nlon + i
                    links[link_seq,0] = visit_writer.quad
                    links[link_seq,1:] = [seq1, seq2, seq3, seq4]
                    link_seq += 1

            self.pts = xyzs.ravel().tolist()
            self.links = links.tolist()

        elif geo == 'plane':
            if ll_type == 'regular':
                self.dimensions = (nlon,nlat,1)

            elif ll_type == 'gaussian':
                self.x = self.lons.tolist()
                self.y = self.lats.tolist()
                self.z = [0]




    def write_with_variables(self, fpath, variables, use_binary=True):
        '''
        variables : (('var1',dimension,centering,array),...)
        dimension : 1 for scalar or 3 for vector
        centering : 0 for cell-wise or 1 for point-wise
        '''
        if self.geo == 'sphere':
            visit_writer.WriteUnstructuredMesh(fpath, use_binary, \
                    self.pts, self.links, variables)

        elif self.geo == 'plane':
            if self.ll_type == 'regular':
                visit_writer.WriteRegularMesh(fpath, use_binary, \
                        self.dimensions, variables)

            elif self.ll_type == 'gaussian':
                visit_writer.WriteRectilinearMesh(fpath, use_binary, \
                        self.x, self.y, self.z, variables)



    def make_vtk_from_netcdf(self, target_fpath, ncf, varname_list):
        variables = list()
        for varname in varname_list:
            print '\t%s'%(varname)
            var = ncf.variables[varname][:]
            assert var.size == self.up_size, '%s size=%d is not same with up_size=%d'%(varname, var.size, self.up_size)

            variables.append( (varname, 1, 1, var.tolist()) )

        self.write_with_variables(target_fpath, variables)




def make_spherical_harmonics(m, n):
    '''
    Generate a VTK file of the Spherical Harmonics using LatlonVTK2D
    '''
    from scipy.special import sph_harm

    nlat, nlon = 180, 360
    ll_type = 'regular'
    ll_vtk = LatlonVTK2D(nlat, nlon, ll_type, 'sphere')

    psi = np.zeros(ll_vtk.nsize, 'f8')
    seq = 0
    for lat in ll_vtk.lats:
        for lon in ll_vtk.lons:
            psi[seq] = sph_harm(m, n, lon, np.pi/2-lat).real
            seq += 1

    variables = (('psi', 1, 1, psi.tolist()),)
    ll_vtk.write_with_variables('spherical_harmonics_m%d_n%d_latlon.vtk'%(m,n), variables)
    #ll_vtk.write_with_variables('spherical_harmonics_m%d_n%d_gaussian.vtk'%(m,n), variables)




if __name__ == '__main__':
    '''
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('nc_fpath', type=str, help='path of the NetCDF file')
    args = parser.parse_args()

    assert os.path.exists(args.nc_fpath), "{} is not found.".format(args.nc_fpath)
    ncf = nc.Dataset(args.nc_fpath, 'r')
    nlat, nlon = ncf.nlat, ncf.nlon
    ll_vtk = LatlonVTK2D(nlat, nlon)

    print 'Generate a VTK file from a NetCDF file'
    print 'nlat=%d, nlon=%d'%(nlat, nlon)
    print 'variables:'

    target_fpath = args.nc_fpath.replace('.nc', '.vtk')
    varname_list = [str(name) for name in ncf.variables.keys()]
    ll_vtk.make_vtk_from_netcdf(target_fpath, ncf, varname_list)
    '''

    make_spherical_harmonics(2,5)
