#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : cube_remap_matrix.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.11.30    start
#             2015.12.2     add LatlonGridRemap class
#             2015.12.10    add bilinear method
#             2015.12.18    generate Voronoi manually without scipy
#             2016.1.18     rename cube_remap.py -> cube_remap_matrix.py
#             2016.1.26     append include_pole option to LatlonGridRemap
#
#
# Description: 
#   Generate a remap_matrix to remap between cubed-sphere and latlon grid
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
from util.convert_coord.cart_ll import latlon2xyz
from util.geometry.voronoi import get_voronoi_scipy, get_voronoi_xyzs




class CubeGridRemap(object):
    def __init__(self, ne, ngq, rotated):
        self.ne = ne
        self.ngq = ngq
        self.rotated = rotated

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
        self.nbrs = cs_ncf.variables['nbrs'][:]              # (up_size,8),gid

        self.rlat = cs_ncf.rlat
        self.rlon = cs_ncf.rlon
        self.nsize = self.up_size

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

        if a <= elems[0]:
            eim = 0
        elif a >= elems[-1]:
            eim = ne-1
        else:
            for eim, (a1,a2) in enumerate( zip(elems[:-1], elems[1:]) ):
                if (a-a1)*(a-a2) <= 0: break

        if b <= elems[0]:
            ejm = 0
        elif b >= elems[-1]:
            ejm = ne-1
        else:
            for ejm, (b1,b2) in enumerate( zip(elems[:-1], elems[1:]) ):
                if (b-b1)*(b-b2) <= 0: break


        return (a,b), (panel,eim+1,ejm+1)



    def get_surround_elem_gids(self, lat, lon):
        '''
        return ngq*ngq uids of surrounding element
        '''
        ne, ngq = self.ne, self.ngq
        uids = self.uids
        ij2gid = self.ij2gid

        (a,b), (panel,ei,ej) = self.get_surround_elem(lat, lon)

        gid_list = list()
        for gj in xrange(1,ngq+1):
            for gi in xrange(1,ngq+1):
                gid = ij2gid[(panel,ei,ej,gi,gj)]
                gid_list.append(gid)

        return np.array(gid_list)



    def get_surround_4_gids(self, lat, lon):
        '''
        return four uids of surrounding box (gid)
        '''

        ne, ngq = self.ne, self.ngq
        uids = self.uids
        alpha_betas = self.alpha_betas
        ij2gid = self.ij2gid

        (a,b), (panel,ei,ej) = self.get_surround_elem(lat, lon)
        gid0 = ij2gid[(panel,ei,ej,1,1)]

        a_list = [alpha_betas[gid0+i][0] for i in xrange(ngq)]
        b_list = [alpha_betas[gid0+i][1] for i in xrange(0,ngq*ngq,ngq)]

        if a <= a_list[0]:
            gim = 0
        elif a >= a_list[-1]:
            gim = ngq-2
        else:
            for gim, (a1,a2) in enumerate( zip(a_list[:-1], a_list[1:]) ):
                if (a-a1)*(a-a2) <= 0: break

        if b <= b_list[0]:
            gjm = 0
        elif b >= b_list[-1]:
            gjm = ngq-2
        else:
            for gjm, (b1,b2) in enumerate( zip(b_list[:-1], b_list[1:]) ):
                if (b-b1)*(b-b2) <= 0: break

        gi, gj = gim+1, gjm+1

        gid0 = ij2gid[(panel,ei,ej,gi  ,gj  )]
        gid1 = ij2gid[(panel,ei,ej,gi+1,gj  )]
        gid2 = ij2gid[(panel,ei,ej,gi  ,gj+1)]
        gid3 = ij2gid[(panel,ei,ej,gi+1,gj+1)]

        return (a,b,panel), (gid0, gid1, gid2, gid3)



    def get_surround_idxs(self, lat, lon):
        '''
        return four uids of surrounding box (uid)
        '''
        abp, gids = self.get_surround_4_gids(lat, lon)

        return [self.uids[gid] for gid in gids]



    def get_neighbors(self, uid):
        gids = self.nbrs[uid,:]

        return self.uids[gids]



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




def make_padded_lats_lons(nlat, nlon, ll_type):
    if 'shift_lon' in ll_type:
        ll_type = ll_type.split('-')[0]
        shift_lon = True
    else:
        ll_type = ll_type
        shift_lon = False

    if shift_lon:
        dlon = 2*np.pi/nlon
        padded_lons = np.linspace(dlon/2, 2*np.pi+dlon/2, nlon+1)
    else:
        padded_lons = np.linspace(0, 2*pi, nlon+1)


    if ll_type == 'include_pole':
        padded_lats = np.linspace(-pi/2, pi/2, nlat)

    elif ll_type == 'regular':
        dlat = np.pi/nlat 
        padded_lats = np.zeros(nlat+2)
        padded_lats[1:-1] = np.linspace((-pi+dlat)/2, (pi-dlat)/2, nlat)
        padded_lats[0] = -np.pi/2
        padded_lats[-1] = np.pi/2

    elif ll_type == 'gaussian':
        import spharm   # NCAR SPHEREPACK
        degs, wts = spharm.gaussian_lats_wts(nlat)
        pts = np.deg2rad(degs[::-1])    # convert to south first
        padded_lats = np.zeros(nlat+2)
        padded_lats[1:-1] = pts
        padded_lats[0] = -np.pi/2
        padded_lats[-1] = np.pi/2

    else:
        raise ValueError, 'Wrong ll_type=%s. Support ll_type: regular, gaussian'%(ll_type)

    return ll_type, padded_lats, padded_lons




def make_lats_lons(nlat, nlon, ll_type):
    ll_type, padded_lats, padded_lons = \
            make_padded_lats_lons(nlat, nlon, ll_type)

    sli = slice(None,None) if ll_type=='include_pole' else slice(1,-1)
    lats = padded_lats[sli]
    lons = padded_lons[:-1]

    return lats, lons




class LatlonGridRemap(object):
    def __init__(self, nlat, nlon, ll_type):
        '''
        Note: The latlon grid should not include the pole.
        Support ll_type: regular, gaussian, include_pole with shift_lon
        '''
        self.nlat = nlat
        self.nlon = nlon
        self.nsize = nlat*nlon

        ll_type, padded_lats, padded_lons = \
                make_padded_lats_lons(nlat, nlon, ll_type)

        latlons = np.zeros((self.nsize,2), 'f8')
        xyzs = np.zeros((self.nsize,3), 'f8')

        sli = slice(None,None) if ll_type=='include_pole' else slice(1,-1)
        seq = 0
        for lat in padded_lats[sli]:
            for lon in padded_lons[:-1]:
                latlons[seq,:] = (lat,lon)
                xyzs[seq,:] = latlon2xyz(lat,lon)
                seq += 1


        self.ll_type = ll_type
        self.grid_type = '%s latlon'%ll_type
        self.padded_lats = padded_lats
        self.padded_lons = padded_lons
        self.dlat = padded_lats[2] - padded_lats[1]
        self.dlon = padded_lons[2] - padded_lons[1]
        self.latlons = latlons
        self.xyzs = xyzs



    def get_surround_idxs(self, lat, lon):
        '''
        return indices of surrounding box
        If the target is near pole, the nearest point is returned.
        '''

        nlat, nlon = self.nlat, self.nlon
        ll_type = self.ll_type
        padded_lats = self.padded_lats
        padded_lons = self.padded_lons

        ljp = np.where( (padded_lats[:-1]<=lat)*(padded_lats[1:]>lat) )[0]
        li = np.where( (padded_lons[:-1]<=lon)*(padded_lons[1:]>lon) )[0]
        
        if ll_type == 'include_pole':
            if lat >= padded_lats[-1]: 
                lj = nlat - 2
            else:
                lj = int(ljp)
        else:
            if lat >= padded_lats[-1]: 
                lj = nlat - 1
            else:
                lj = int(ljp) - 1


        if lon >= padded_lons[-1] or lon <= padded_lons[0]:
            li = nlon-1
        else:
            li = int(li)


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
        padded_lats, padded_lons = self.padded_lats, self.padded_lons

        lj = idx//nlon
        li = idx%nlon

        lat = padded_lats[lj+1]
        lon = padded_lons[li]

        lat1, lat2 = lat-dlat/2, lat+dlat/2
        lon1 = lon - dlon/2 if li != 0 else 2*pi - dlon/2
        lon2 = lon + dlon/2

        if lj == 0:
            ll_vertices = [(-pi/2,0), (lat2,lon2), (lat2,lon1)]

        elif lj == nlat-1:
            ll_vertices = [(pi/2,0), (lat1,lon1), (lat1,lon2)]

        else:
            ll_vertices = [(lat1,lon1), (lat1,lon2), (lat2,lon2), (lat2,lon1)]

        return [latlon2xyz(*ll) for ll in ll_vertices]




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
            choices=['regular', 'gaussian', 'include_pole', 'regular-shift_lon'])
    parser.add_argument('direction', type=str, help='remap direction', \
            choices=['ll2cs','cs2ll'])
    parser.add_argument('method', type=str, help='remap method', \
            choices=['bilinear', 'vgecore', 'rbf', 'lagrange'])
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

        #yn = raw_input('Continue (Y/n)? ')
        #if yn.lower() == 'n':
        #    sys.exit()

        if not os.path.exists(output_dir):
            print "%s is not found. Make output directory."%(output_dir)
            os.makedirs(output_dir)

        #if os.path.exists(output_fpath):
        #    yn = raw_input("%s is found. Overwrite(Y/n)? "%output_fpath)
        #    if yn.lower() == 'n':
        #        sys.exit()

    comm.Barrier()


    if myrank ==0:
        #------------------------------------------------------------
        print 'Prepare to save as NetCDF'
        #------------------------------------------------------------
        ncf = nc.Dataset(output_fpath, 'w', format='NETCDF3_CLASSIC')
        ncf.description = 'Remapping between Cubed-sphere and Latlon grids'
        ncf.remap_method = method
        ncf.remap_direction = direction

 
    #-------------------------------------------------
    if myrank == 0: print 'Make a remap matrix'
    #-------------------------------------------------
    cs_obj = CubeGridRemap(ne, ngq, rotated)
    ll_obj = LatlonGridRemap(nlat, nlon, ll_type)

    if method == 'bilinear':
        from cube_remap_bilinear import Bilinear
        rmp = Bilinear(cs_obj, ll_obj, direction)
        src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    elif method == 'vgecore':
        from cube_remap_vgecore import VGECoRe
        rmp = VGECoRe(cs_obj, ll_obj, direction)
        dst_address, src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    elif method == 'rbf':
        from cube_remap_rbf import RadialBasisFunction
        rmp = RadialBasisFunction(cs_obj, ll_obj, direction)
        src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    elif method == 'lagrange':
        assert direction=='cs2ll', "Lagrange method supports only 'cs2ll'"
        from cube_remap_lagrange import LagrangeBasisFunction
        rmp = LagrangeBasisFunction(cs_obj, ll_obj)
        src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    else:
        raise ValueError, 'The remap method %s is not supported yet.'%(method)


    if myrank ==0:
        #------------------------------------------------------------
        print 'Save as NetCDF'
        #------------------------------------------------------------
        ncf.rotated = str(cs_obj.rotated).lower()
        ncf.ne = cs_obj.ne
        ncf.ngq = cs_obj.ngq
        ncf.ep_size = cs_obj.ep_size
        ncf.up_size = cs_obj.up_size
        ncf.nlat = ll_obj.nlat
        ncf.nlon = ll_obj.nlon
        ncf.ll_size = ll_obj.nsize

        if method == 'vgecore':
            rmp.set_netcdf_remap_matrix(ncf, dst_address, src_address, remap_matrix)
        else:
            rmp.set_netcdf_remap_matrix(ncf, src_address, remap_matrix)

        ncf.close()
        print 'Done.'
