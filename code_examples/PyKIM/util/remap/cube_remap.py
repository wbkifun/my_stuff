#------------------------------------------------------------------------------
# filename  : cube_remap.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.30    start
#             2015.12.2     add LatlonGridRemap class
#             2015.12.10    add bilinear method
#             2015.12.18    generate Voronoi manually without scipy
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
# 
# Class:
#   CubeGridRemap
#   LatlonGridRemap
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from numpy import pi
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from util.grid.path import dir_cs_grid
from util.convert_coord.cs_ll import latlon2abp
from util.geometry.voronoi import get_voronoi_scipy, get_voronoi_xyzs




class CubeGridRemap(object):
    def __init__(self, ne, ngq, rotated):
        self.ne = ne
        self.ngq = ngq
        self.rotated = rotated

        if rotated:
            #korea centered
            self.rlon = np.deg2rad(38)
            self.rlat = np.deg2rad(127)
        else:
            self.rlon = 0
            self.rlat = 0

        self.grid_type = 'cubed-sphere'


        #-----------------------------------------------------
        # Read the grid indices
        #-----------------------------------------------------
        if rotated:
            cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d_rotated.nc'%(ne, ngq)
        else:
            cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)

        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')

        self.ep_size = len( cs_ncf.dimensions['ep_size'] )
        self.up_size = len( cs_ncf.dimensions['up_size'] )
        self.gq_indices = cs_ncf.variables['gq_indices'][:]  # (ep_size,5)
        self.is_uvps = cs_ncf.variables['is_uvps'][:]        # (ep_size)
        self.uids = cs_ncf.variables['uids'][:]              # (ep_size)
        self.gids = cs_ncf.variables['gids'][:]              # (up_size)
        self.xyzs = cs_ncf.variables['xyzs'][:]              # (up_size,3)
        self.latlons = cs_ncf.variables['latlons'][:]        # (up_size,2)
        self.alpha_betas = cs_ncf.variables['alpha_betas'][:]# (ep_size,2)

        #self.mvps = cs_ncf.variables['mvps'][:]              # (ep_size,4)
        self.nbrs = cs_ncf.variables['nbrs'][:]              # (up_size,8)

        self.ij2gid = dict()
        for gid, ij in enumerate(self.gq_indices):   # ij: (panel,ei,ej,gi,gj)
            self.ij2gid[tuple(ij)] = gid


        
    def get_surround_elem(self, lat, lon):
        '''
        return a left-bottom GQ point of surrounding element
        '''
        ne, ngq = self.ne, self.ngq
        rlat, rlon = self.rlat, self.rlon
        ij2gid = self.ij2gid

        abp_dict = latlon2abp(lat, lon, rlat, rlon)
        panel = min(abp_dict.keys())
        a, b = abp_dict[panel]


        elems = np.linspace(-pi/4, pi/4, ne+1)

        for eim, (a1,a2) in enumerate( zip(elems[:-1], elems[1:]) ):
            if (a-a1)*(a-a2) <= 0: break

        for ejm, (b1,b2) in enumerate( zip(elems[:-1], elems[1:]) ):
            if (b-b1)*(b-b2) <= 0: break


        return (a,b), (panel,eim+1,ejm+1)



    def get_surround_elem_uids(self, lat, lon):
        '''
        return ngq*ngq uids of surrounding element
        '''
        ne, ngq = self.ne, self.ngq
        uids = self.uids
        ij2gid = self.ij2gid

        (a,b), (panel,ei,ej) = self.get_surround_elem(lat, lon)

        uid_list = list()
        for gj in xrange(1,ngq+1):
            for gi in xrange(1,ngq+1):
                uid = uids[ ij2gid[(panel,ei,ej,gi,gj)] ]
                uid_list.append(uid)

        return uid_list



    def get_surround_4_uids(self, lat, lon):
        '''
        return four uids of surrounding box
        '''

        ne, ngq = self.ne, self.ngq
        uids = self.uids
        alpha_betas = self.alpha_betas
        ij2gid = self.ij2gid

        (a,b), (panel,ei,ej) = self.get_surround_elem(lat, lon)
        gid0 = ij2gid[(panel,ei,ej,1,1)]

        a_list = [alpha_betas[gid0+i][0] for i in xrange(ngq)]
        b_list = [alpha_betas[gid0+i][1] for i in xrange(0,ngq*ngq,ngq)]

        for gim, (a1,a2) in enumerate( zip(a_list[:-1], a_list[1:]) ):
            if (a-a1)*(a-a2) <= 0: break

        for gjm, (b1,b2) in enumerate( zip(b_list[:-1], b_list[1:]) ):
            if (b-b1)*(b-b2) <= 0: break

        gi, gj = gim+1, gjm+1

        uid0 = uids[ ij2gid[(panel,ei,ej,gi  ,gj  )] ]
        uid1 = uids[ ij2gid[(panel,ei,ej,gi+1,gj  )] ]
        uid2 = uids[ ij2gid[(panel,ei,ej,gi  ,gj+1)] ]
        uid3 = uids[ ij2gid[(panel,ei,ej,gi+1,gj+1)] ]

        return (uid0, uid1, uid2, uid3)



    def get_voronoi_scipy(self, uid):
        '''
        Try scipy.spatial.Voronoi, but wrong result
        '''
        nbrs = [gid for gid in self.nbrs[uid] if gid != -1]

        lat0, lon0 = self.latlons[uid]
        nbr_latlons = [self.latlons[self.uids[gid]] for gid in nbrs]
        xy_vertices, vor_obj = get_voronoi_scipy(lat0, lon0, nbr_latlons)
        return xy_vertices, vor_obj



    def get_voronoi(self, uid):
        nbrs = [gid for gid in self.nbrs[uid] if gid != -1]

        xyz0 = self.xyzs[uid]
        nbr_xyzs = [self.xyzs[self.uids[gid]] for gid in nbrs]
        voronoi_xyzs = get_voronoi_xyzs(xyz0, nbr_xyzs)

        return voronoi_xyzs




class LatlonGridRemap(object):
    def __init__(self, nlat, nlon, latlon_type='regular'):
        '''
        Note: The latlon grid should not include the pole.
        Support latlon_type: regular, gaussian
        '''
        self.nlat = nlat
        self.nlon = nlon
        self.latlon_type = latlon_type
        self.grid_type = '%s latlon'%latlon_type

        self.nsize = nlat*nlon
        self.tmp_lons = np.linspace(0, 2*pi, nlon+1)

        if latlon_type == 'regular':
            self.tmp_lats = np.linspace(-pi/2, pi/2, nlat+2)

        elif latlon_type == 'gaussian':
            import spharm   # NCAR SPHEREPACK
            degs, wts = spharm.gaussian_lats_wts(nlat)
            pts = np.deg2rad(degs[::-1])    # convert to south first
            self.tmp_lats = np.zeros(nlat+2)
            self.tmp_lats[1:-1] = pts
            self.tmp_lats[0] = -np.pi/2
            self.tmp_lats[-1] = np.pi/2

        else:
            raise ValueError, 'Wrong latlon_type=%s. Support latlon_type: regular, gaussian'%(latlon_type)


        self.latlons = np.zeros((self.nsize,2), 'f8')
        seq = 0
        for lat in self.tmp_lats[1:-1]:
            for lon in self.tmp_lons[:-1]:
                self.latlons[seq,:] = (lat,lon)
                seq += 1

        self.dlat = self.tmp_lats[2] - self.tmp_lats[1]
        self.dlon = self.tmp_lons[2] - self.tmp_lons[1]



    def get_surround_idxs(self, lat, lon):
        '''
        return indices of surrounding box
        If the target is near pole, the nearest point is returned.
        '''

        nlat, nlon = self.nlat, self.nlon
        tmp_lats = self.tmp_lats
        tmp_lons = self.tmp_lons

        ljp = np.where( (tmp_lats[:-1]<=lat)*(tmp_lats[1:]>lat) )[0]
        li = np.where( (tmp_lons[:-1]<=lon)*(tmp_lons[1:]>lon) )[0]

        if lat >= tmp_lats[-1]: ljp = nlat-1
        else: lj = int(ljp) - 1
        if lon >= tmp_lons[-1]: li = nlon-1
        else: li = int(li)

        try:
            lj = int(ljp) - 1
            li = int(li)
        except Exception as e:
            print ''
            print 'lat,lon', lat, lon
            print e.message
            import sys
            sys.exit()

        if lj == -1:
            return (li, -1, -1, -1)

        elif lj == nlat-1:
            return (lj*nlon + li, -1, -1, -1)

        elif li == nlon-1:
            idx0 = (lj  )*nlon + (li  )
            idx1 = (lj  )*nlon + (li+1-nlon)
            idx2 = (lj+1)*nlon + (li  )
            idx3 = (lj+1)*nlon + (li+1-nlon)

            return (idx0, idx1, idx2, idx3)

        else:
            idx0 = (lj  )*nlon + (li  )
            idx1 = (lj  )*nlon + (li+1)
            idx2 = (lj+1)*nlon + (li  )
            idx3 = (lj+1)*nlon + (li+1)

            return (idx0, idx1, idx2, idx3)



    def get_neighbors(self, idx):
        nlat, nlon = self.nlat, self.nlon

        i = idx%nlon
        j = idx//nlon

        im = i-1 if i != 0 else i+nlon-1
        ip = i+1 if i != nlon-1 else i-nlon+1

        if j == 0:
            ijs = ((im,j  ),          (ip,j  ), \
                   (im,j+1), (i,j+1), (ip,j+1))

        elif j == nlat-1:
            ijs = ((im,j-1), (i,j-1), (ip,j-1), \
                   (im,j  ),          (ip,j  ))

        else:
            ijs = ((im,j-1), (i,j-1), (ip,j-1), \
                   (im,j  ),          (ip,j  ), \
                   (im,j+1), (i,j+1), (ip,j+1))

        return [j*nlon+i for (i,j) in ijs]



    def get_voronoi(self, idx):
        nlat, nlon = self.nlat, self.nlon
        dlat, dlon = self.dlat, self.dlon
        tmp_lats, tmp_lons = self.tmp_lats, self.tmp_lons

        lj = idx//nlon
        li = idx%nlon

        lat = tmp_lats[lj+1]
        lon = tmp_lons[li]

        lat1, lat2 = lat-dlat/2, lat+dlat/2
        lon1, lon2 = lon-dlon/2, lon+dlon/2

        if lj == 0:
            return (-pi/2,0), (lat2,lon2), (lat2,lon1)

        elif lj == nlat-1:
            return (pi/2,0), (lat1,lon1), (lat1,lon2)

        else:
            return (lat1,lon1), (lat1,lon2), (lat2,lon2), (lat2,lon1)




def save_netcdf_remap_matrix(cs_obj, ll_obj, fpath, direction, method):
    nproc = comm.Get_size()
    myrank = comm.Get_rank()


    #------------------------------------------------------------
    if myrank == 0: print 'Make dsw_dict'
    #------------------------------------------------------------
    if method == 'bilinear':
        from cube_remap_bilinear import Bilinear
        remap = Bilinear(cs_obj, ll_obj)

    elif method == 'vgecore':
        from cube_remap_vgecore import VGECoRe
        remap = VGECoRe(cs_obj, ll_obj)

    else:
        raise ValueError, 'The remap method %s is not supported yet.'%(method)

    if nproc == 1:
        dsw_dict = getattr(remap, 'make_dsw_dict_%s'%direction)()
    else:
        dsw_dict = getattr(remap, 'make_dsw_dict_%s_mpi'%direction)()


    if myrank == 0:
        #------------------------------------------------------------
        print 'Make remap sparse matrix from the dsw_dict'
        #------------------------------------------------------------
        num_links = sum( [len(sw_list) for sw_list in dsw_dict.values()] )
        dst_address = np.zeros(num_links, 'i4')
        src_address = np.zeros(num_links, 'i4')
        remap_matrix = np.zeros(num_links, 'f8')

        seq = 0
        for dst in sorted(dsw_dict.keys()):
            for src, wgt in sorted(dsw_dict[dst]):
                dst_address[seq] = dst
                src_address[seq] = src
                remap_matrix[seq] = wgt
                seq += 1


        #------------------------------------------------------------
        print 'Save as NetCDF'
        #------------------------------------------------------------
        ncf = nc.Dataset(fpath, 'w', format='NETCDF3_CLASSIC') # for pnetcdf
        ncf.description = 'Remapping between Cubed-sphere and Latlon grids'
        ncf.remap_method = method
        ncf.remap_direction = direction

        ncf.rotated = str(cs_obj.rotated).lower()
        ncf.ne = cs_obj.ne
        ncf.ngq = cs_obj.ngq
        ncf.ep_size = cs_obj.ep_size
        ncf.up_size = cs_obj.up_size
        ncf.nlat = ll_obj.nlat
        ncf.nlon = ll_obj.nlon
        ncf.ll_size = ll_obj.nsize

        ncf.createDimension('num_links', num_links)

        vdst_address = ncf.createVariable('dst_address', 'i4', ('num_links',))
        vsrc_address = ncf.createVariable('src_address', 'i4', ('num_links',))
        vremap_matrix = ncf.createVariable('remap_matrix', 'f8', ('num_links',))

        vdst_address[:] = dst_address[:]
        vsrc_address[:] = src_address[:]
        vremap_matrix[:] = remap_matrix[:]

        ncf.close()




if __name__ == '__main__':
    import argparse
    import os
    import sys
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myrank = comm.Get_rank()


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rotated', action='store_true', help='Korea centered rotation')
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('nlat_nlon', type=str, help='(nlat)x(nlon)')
    parser.add_argument('ll_type', type=str, help='latlon grid type', \
            choices=['regular', 'gaussian'])
    parser.add_argument('direction', type=str, help='remap direction', \
            choices=['ll2cs','cs2ll'])
    parser.add_argument('method', type=str, help='remap method', \
            choices=['bilinear', 'vgecore', 'lagrange'])
    parser.add_argument('output_dir', nargs='?', type=str, \
            help='output directory', default='./remap_matrix/')
    args = parser.parse_args()


    rotated = args.rotated
    ne, ngq = args.ne, 4
    cs_type = 'rotated' if rotated else 'regular'
    nlat, nlon = [int(n) for n in args.nlat_nlon.split('x')]
    ll_type = args.ll_type
    direction = args.direction
    method = args.method
    output_dir = args.output_dir
    output_fname = 'remap_%s_ne%d_%s_%dx%d_%s_%s.nc'%(direction, ne, cs_type, nlat, nlon, ll_type, method)
    output_fpath = output_dir + output_fname


    #-------------------------------------------------
    # check
    #-------------------------------------------------
    if myrank == 0:
        if direction == 'll2cs':
            print 'source grid: latlon(%s), nlat=%d, nlon=%d'%(ll_type, nlat, nlon)
            print 'target grid: cubed-sphere(%s), ne=%d, ngq=%d'%(cs_type, ne, ngq)

        elif direction == 'cs2ll':
            print 'source grid: cubed-sphere(%s), ne=%d, ngq=%d'%(cs_type, ne, ngq)
            print 'target grid: latlon(%s), nlat=%d, nlon=%d'%(ll_type, nlat, nlon)


        print 'remap method: %s'%method
        print 'output directory: %s'%output_dir
        print 'output filename: %s'%output_fname

        yn = raw_input('Continue (Y/n)? ')
        if yn.lower() == 'n':
            sys.exit()

        if not os.path.exists(output_dir):
            print "%s is not found. Make output directory."%(output_dir)
            os.makedirs(output_dir)

        if os.path.exists(output_fpath):
            yn = raw_input("%s is found. Overwrite(Y/n)? "%output_fpath)
            if yn.lower() == 'n':
                sys.exit()

    comm.Barrier()


    #-------------------------------------------------
    # Generate a NetCDF file of remap matrix
    #-------------------------------------------------
    cs_obj = CubeGridRemap(ne, ngq, rotated)
    ll_obj = LatlonGridRemap(nlat, nlon, ll_type)

    save_netcdf_remap_matrix(cs_obj, ll_obj, output_fpath, direction, method)
