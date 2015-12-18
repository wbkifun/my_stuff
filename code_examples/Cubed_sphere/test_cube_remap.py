#------------------------------------------------------------------------------
# filename  : test_cube_remap.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.4   start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_get_surround_elem_uids():
    '''
    CubeGridRemap.get_surround_elem_uids(): ne=30
    '''
    from cube_remap import CubeGridRemap
    from pkg.convert_coord.cs_ll import abp2latlon


    ne, ngq = 30, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)
    td = (np.pi/2)/ne/3/2   # tiny delta

    ij = (1,2,1,1,1)
    gid = cube.ij2gid[ij]
    alpha, beta = cube.alpha_betas[gid]
    lat, lon = abp2latlon(alpha+td, beta+td, ij[0])
    ret_uids = cube.get_surround_elem_uids(lat, lon)
    a_equal(ret_uids, [3,16,17,18,7,19,20,21,11,22,23,24,15,25,26,27])




def test_get_surround_4_uids():
    '''
    CubeGridRemap.get_surround_4_uids(): ne=30
    '''
    from cube_remap import CubeGridRemap
    from pkg.convert_coord.cs_ll import abp2latlon


    ne, ngq = 30, 4
    rotated = False

    cube = CubeGridRemap(ne, ngq, rotated)
    td = (np.pi/2)/ne/3/2   # tiny delta

    ij = (1,2,1,1,1)
    gid = cube.ij2gid[ij]
    alpha, beta = cube.alpha_betas[gid]
    lat, lon = abp2latlon(alpha+td, beta+td, ij[0])
    ret_uids = cube.get_surround_4_uids(lat, lon)
    a_equal(ret_uids, [3,16,7,19])

    ij = (1,2,1,2,3)
    gid = cube.ij2gid[ij]
    alpha, beta = cube.alpha_betas[gid]
    lat, lon = abp2latlon(alpha+td, beta+td, ij[0])
    ret_uids = cube.get_surround_4_uids(lat, lon)
    a_equal(ret_uids, [22,23,25,26])


    '''
    for (alpha,beta), ij in zip(cube.alpha_betas, cube.gq_indices):
        panel, ei, ej, gi, gj = ij

        if (gi < ngq) and (gj < ngq):
            lat, lon = abp2latlon(alpha+td, beta+td, panel)
            ret_ij = cube.get_surround_ij(lat, lon)
            a_equal(ij, ret_ij)
    '''




def test_cs_get_voronoi():
    '''
    CubeGridRemap.get_voronoi(): ne=30, some points
    '''
    from cube_remap import CubeGridRemap
    from scipy.spatial import voronoi_plot_2d
    import matplotlib.pyplot as plt

    ne, ngq = 30, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)

    uid = 0
    xy_vertices, vor = cube.get_voronoi(uid)
    expect = [( 4.54850726e-03, 3.80070853e-05), \
              ( 2.30716873e-03, 3.92011929e-03), \
              (-2.30716873e-03, 3.92011929e-03), \
              (-4.54850726e-03, 3.80070853e-05), \
              (-2.24133853e-03,-3.95812638e-03), \
              ( 2.24133853e-03,-3.95812638e-03)]
    aa_equal(expect, xy_vertices, 10)

    uid = 1
    xy_vertices, vor = cube.get_voronoi(uid)
    expect = [( 5.98890285e-03,-2.13793346e-03), \
              ( 5.01646021e-03, 3.93916864e-03), \
              (-2.27448976e-03, 3.93916864e-03), \
              (-4.54802687e-03,-7.61894981e-05), \
              (-7.97052613e-04,-6.32824066e-03), \
              ( 4.91440620e-03,-4.03563236e-03)]
    aa_equal(expect, xy_vertices, 10)




def plot_cs_voronoi():
    '''
    CubeGridRemap.get_voronoi(): Plot
    '''
    from cube_remap import CubeGridRemap
    from scipy.spatial import voronoi_plot_2d
    import matplotlib.pyplot as plt

    #ne, ngq = 30, 4
    ne, ngq = 3, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)

    print 'ne=%d, ngq=%d'%(ne, ngq)

    for uid in xrange(20):
        xy_vertices, vor = cube.get_voronoi(uid)

        print ''
        print 'uid', uid
        print 'vor.vertices\n', vor.vertices
        print 'vor.ridge_points\n', vor.ridge_points
        print 'vor.ridge_vertices\n', vor.ridge_vertices
        print ''
        print 'xy_vertices\n', xy_vertices
        voronoi_plot_2d(vor)
        #plt.savefig('voronoi_%d.png'%uid)
        plt.show()




def test_cs_get_voronoi_area():
    '''
    CubeGridRemap.get_voronoi(): check the sphere area, ne=3
    '''
    from cube_remap import CubeGridRemap
    from pkg.convert_coord.cart_cs import xyp2xyz
    from pkg.convert_coord.cart_rotate import xyz_rotate_reverse
    from area_sphere import area_polygon_sphere
    from math import fsum, pi

    ne, ngq = 3, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)

    area_list = list()
    for uid in xrange(cube.up_size):
        xy_list, vor = cube.get_voronoi(uid)
        xyzs = [xyp2xyz(x,y,1) for x,y in xy_list]
        area_list.append( area_polygon_sphere(xyzs) )

    aa_equal(fsum(area_list), 4*pi, 15)




def test_get_surround_idxs():
    '''
    LatlonGridRemap.get_surround_idxs(): nlat=180, nlon=360 (regular)
    '''
    from cube_remap import LatlonGridRemap


    nlat, nlon = 180, 360
    tx = 1e-3
    ll = LatlonGridRemap(nlat, nlon, 'regular')

    # Near south pole
    lat0 = ll.tmp_lats[1]
    lon0 = ll.tmp_lons[7]
    ret_idxs = ll.get_surround_idxs(lat0-tx,lon0+tx)
    equal(ret_idxs, (7,-1,-1,-1))

    # Near north pole
    lat0 = ll.tmp_lats[-2]
    lon0 = ll.tmp_lons[-2]
    ret_idxs = ll.get_surround_idxs(lat0+tx,lon0+tx)
    equal(ret_idxs, (nlat*nlon-1,-1,-1,-1))

    # First box
    lat0 = ll.tmp_lats[1]
    lon0 = ll.tmp_lons[0]
    ret_idxs = ll.get_surround_idxs(lat0+tx,lon0+tx)
    a_equal(ret_idxs, [0,1,nlon,nlon+1])

    # Last box
    lat0 = ll.tmp_lats[-2]
    lon0 = ll.tmp_lons[-2]
    ret_idxs = ll.get_surround_idxs(lat0-tx,lon0+tx)
    a_equal(ret_idxs, [(nlat-1)*nlon-1, (nlat-2)*nlon, nlat*nlon-1, (nlat-1)*nlon])

    # Near Meridian
    lat0 = ll.tmp_lats[1]
    lon0 = ll.tmp_lons[-2]
    ret_idxs = ll.get_surround_idxs(lat0+tx,lon0+tx)
    a_equal(ret_idxs, [nlon-1,0,nlon*2-1,nlon])

    # Error cases
    lat, lon = -0.785398163397, 6.28318530718
    ret_idxs = ll.get_surround_idxs(lat,lon)
    a_equal(ret_idxs, [16199, 15840, 16559, 16200])




def test_get_surround_idxs_2():
    '''
    LatlonGridRemap.get_surround_idxs(): nlat=192, nlon=384 (gaussian)
    '''
    from cube_remap import LatlonGridRemap


    nlat, nlon = 192, 384
    tx = 1e-5
    ll = LatlonGridRemap(nlat, nlon, 'gaussian')

    # Near south pole
    lat0 = ll.tmp_lats[1]
    lon0 = ll.tmp_lons[7]
    ret_idxs = ll.get_surround_idxs(lat0-tx,lon0+tx)
    equal(ret_idxs, (7,-1,-1,-1))

    # Near north pole
    lat0 = ll.tmp_lats[-2]
    lon0 = ll.tmp_lons[-2]
    ret_idxs = ll.get_surround_idxs(lat0+tx,lon0+tx)
    equal(ret_idxs, (nlat*nlon-1,-1,-1,-1))

    # First box
    lat0 = ll.tmp_lats[1]
    lon0 = ll.tmp_lons[0]
    ret_idxs = ll.get_surround_idxs(lat0+tx,lon0+tx)
    a_equal(ret_idxs, [0,1,nlon,nlon+1])

    # Last box
    lat0 = ll.tmp_lats[-2]
    lon0 = ll.tmp_lons[-2]
    ret_idxs = ll.get_surround_idxs(lat0-tx,lon0+tx)
    a_equal(ret_idxs, [(nlat-1)*nlon-1, (nlat-2)*nlon, nlat*nlon-1, (nlat-1)*nlon])

    # Near Meridian
    lat0 = ll.tmp_lats[1]
    lon0 = ll.tmp_lons[-2]
    ret_idxs = ll.get_surround_idxs(lat0+tx,lon0+tx)
    a_equal(ret_idxs, [nlon-1,0,nlon*2-1,nlon])




def test_get_neighbors_latlon():
    '''
    LatlonGridRemap.get_neighbors(): nlat=180, nlon=360 (regular)
    '''
    from cube_remap import LatlonGridRemap

    nlat, nlon = 180, 360
    ll = LatlonGridRemap(nlat, nlon, 'regular')

    ret = ll.get_neighbors(nlon+1)
    expect = (0, 1, 2, nlon, nlon+2, 2*nlon, 2*nlon+1, 2*nlon+2)
    a_equal(expect, ret)

    ret = ll.get_neighbors(nlon)
    expect = (nlon-1, 0, 1, 2*nlon-1, nlon+1, 3*nlon-1, 2*nlon, 2*nlon+1)
    a_equal(expect, ret)

    ret = ll.get_neighbors(1)
    expect = (0, 2, nlon, nlon+1, nlon+2)
    a_equal(expect, ret)

    ret = ll.get_neighbors(0)
    expect = (nlon-1, 1, 2*nlon-1, nlon, nlon+1)
    a_equal(expect, ret)

    ret = ll.get_neighbors((nlat-1)*nlon)
    expect = ((nlat-1)*nlon-1, (nlat-2)*nlon, (nlat-2)*nlon+1, nlat*nlon-1, (nlat-1)*nlon+1)
    a_equal(expect, ret)




def test_get_voronoi_latlon():
    '''
    LatlonGridRemap.get_voronoi(): nlat=180, nlon=360 (regular)
    '''
    from cube_remap import LatlonGridRemap

    nlat, nlon = 180, 360
    ll = LatlonGridRemap(nlat, nlon, 'regular')

    ret = ll.get_voronoi(1)
    expect = [(-1.5707963267948966, 0                   ), \
              (-1.5447610285607269, 0.026179938779914945), \
              (-1.5447610285607269, 0.008726646259971647)]
    aa_equal(expect, ret, 10)

    ret = ll.get_voronoi(nlon)
    expect = [(-1.5447610285607269,-0.00872664625997164), \
              (-1.5447610285607269, 0.008726646259971647), \
              (-1.5274041630712807, 0.008726646259971647), \
              (-1.5274041630712807,-0.00872664625997164)]
    aa_equal(expect, ret, 10)




def exp_remap_sph():
    '''
    Remapping experiment, Spherical harmonics
    '''
    from cube_remap import CubeGridRemap, LatlonGridRemap
    import netCDF4 as nc
    from cube_vtk import CubeVTK2D
    from latlon_vtk import LatlonVTK2D
    from scipy.special import sph_harm
    from standard_errors import sem_1_2_inf


    #----------------------------------------------------------
    # Setup
    #----------------------------------------------------------
    method = 'vgecore'      # 'bilinear', 'vgecore', 'lagrange'
    direction = 'll2cs'
    cs_type = 'regular'     # 'regular', 'rotated'
    ll_type = 'regular'     # 'regular', 'gaussian'

    #ne, ngq = 15, 4
    ne, ngq = 30, 4
    #ne, ngq = 60, 4
    #ne, ngq = 120, 4
    rotated = cs_type == 'rotated'
    cs_obj = CubeGridRemap(ne, ngq, rotated)

    #nlat, nlon = 90, 180
    nlat, nlon = 180, 360
    #nlat, nlon = 360, 720
    #nlat, nlon = 720, 1440

    #nlat, nlon = 192, 384
    ll_obj = LatlonGridRemap(nlat, nlon, ll_type)


    m, n = 16, 32
    ll_f = np.zeros(ll_obj.nsize, 'f8')
    for i, (lat,lon) in enumerate(ll_obj.latlons):
        ll_f[i] = sph_harm(m, n, lon, np.pi/2-lat).real


    print ''
    print 'ne=%d, ngq=%d, %s'%(ne, ngq, cs_type)
    print 'nlat=%d, nlon=%d, %s'%(nlat, nlon, ll_type)
    print 'method: %s'%(method)
    print 'direction: %s'%(direction)
    print 'SPH m=%d, n=%d'%(m, n)


    #----------------------------------------------------------
    # Remap
    #----------------------------------------------------------
    cs_f = np.zeros(cs_obj.up_size, ll_f.dtype)

    remap_dir = '/nas2/user/khkim/remap_matrix/'
    fname = 'remap_%s_ne%d_%s_%dx%d_%s_%s.nc'%(direction, ne, cs_type, nlat, nlon, ll_type, method)

    ncf = nc.Dataset(remap_dir+fname, 'r', 'NETCDF3_CLASSIC')
    num_links = len( ncf.dimensions['num_links'] )
    dsts = ncf.variables['dst_address'][:]
    srcs = ncf.variables['src_address'][:]
    wgts = ncf.variables['remap_matrix'][:]

    for dst, src, wgt in zip(dsts, srcs, wgts):
        cs_f[dst] += ll_f[src]*wgt


    #----------------------------------------------------------
    # Standard errors
    #----------------------------------------------------------
    ref_cs_f = np.zeros_like(cs_f)
    for i, (lat,lon) in enumerate(cs_obj.latlons):
        ref_cs_f[i] = sph_harm(m, n, lon, np.pi/2-lat).real

    L1, L2, Linf = sem_1_2_inf(ref_cs_f, cs_f)
    print ''
    print 'L1', L1
    print 'L2', L2
    print 'Linf', Linf


    #----------------------------------------------------------
    # Plot with vtk
    #----------------------------------------------------------
    vtk_dir = '/nas/scteam/VisIt_data/remap/'
    fpath = vtk_dir + '%s/sph%d%d_%s_ne%d_%s_%dx%d_%s_%s.nc'%(method, m, n, direction, ne, cs_type, nlat, nlon, ll_type, method)

    ll_vtk = LatlonVTK2D(nlat, nlon, ll_type, 'sphere')
    vll = (('ll_f', 1, 1, ll_f.tolist()),)
    ll_vtk.write_with_variables(vtk_dir+'sph%d%d_ll_%dx%d_%s.vtk'%(m,n,nlat,nlon,ll_type), vll)

    cs_vtk = CubeVTK2D(ne, ngq, rotated)
    vcs = (('ref_cs_f', 1, 1, ref_cs_f.tolist()), ('cs_f', 1, 1, cs_f.tolist()))
    cs_vtk.write_with_variables(fpath, vcs)
