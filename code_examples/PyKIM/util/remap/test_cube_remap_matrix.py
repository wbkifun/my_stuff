#------------------------------------------------------------------------------
# filename  : test_cube_remap_matrix.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.4   start
#             2016.1.26   rename test_cube_remap.py to test_cube_remap_matrix.py
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_get_surround_elem_rotated():
    '''
    CubeGridRemap.get_surround_elem() rotated: ne=30
    '''
    from cube_remap import CubeGridRemap
    from util.convert_coord.cs_ll import abp2latlon


    ne, ngq = 30, 4
    rotated = True
    cube = CubeGridRemap(ne, ngq, rotated)

    lat, lon = np.deg2rad(38), np.deg2rad(127)
    (a,b), (panel,ei,ej) = cube.get_surround_elem(lat, lon)
    aa_equal((a,b), (0,0), 15)
    a_equal((panel,ei,ej), (1,15,16))




def test_get_surround_elem_gids():
    '''
    CubeGridRemap.get_surround_elem_gids(): ne=30
    '''
    from cube_remap import CubeGridRemap
    from util.convert_coord.cs_ll import abp2latlon


    ne, ngq = 30, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)
    td = (np.pi/2)/ne/3/2   # tiny delta

    ij = (1,2,1,1,1)
    gid = cube.ij2gid[ij]
    alpha, beta = cube.alpha_betas[gid]
    lat, lon = abp2latlon(alpha+td, beta+td, ij[0])
    ret_gids = cube.get_surround_elem_gids(lat, lon)
    a_equal(ret_gids, [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
    ret_uids = [cube.uids[gid] for gid in ret_gids]
    a_equal(ret_uids, [3,16,17,18,7,19,20,21,11,22,23,24,15,25,26,27])




def test_get_surround_4_gids():
    '''
    CubeGridRemap.get_surround_4_gids(): ne=30
    '''
    from cube_remap import CubeGridRemap

    ne, ngq = 30, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)

    lat, lon = -0.78973737977, 0.0
    abp, ret_gids = cube.get_surround_4_gids(lat, lon)
    a_equal(ret_gids, [71754,71755,71758,71759])




def test_get_surround_idxs_cube():
    '''
    CubeGridRemap.get_surround_idxs(): ne=30
    '''
    from cube_remap import CubeGridRemap
    from util.convert_coord.cs_ll import abp2latlon


    ne, ngq = 30, 4
    rotated = False

    cube = CubeGridRemap(ne, ngq, rotated)
    td = (np.pi/2)/ne/3/2   # tiny delta

    ij = (1,2,1,1,1)
    gid = cube.ij2gid[ij]
    alpha, beta = cube.alpha_betas[gid]
    lat, lon = abp2latlon(alpha+td, beta+td, ij[0])
    ret_uids = cube.get_surround_idxs(lat, lon)
    a_equal(ret_uids, [3,16,7,19])

    ij = (1,2,1,2,3)
    gid = cube.ij2gid[ij]
    alpha, beta = cube.alpha_betas[gid]
    lat, lon = abp2latlon(alpha+td, beta+td, ij[0])
    ret_uids = cube.get_surround_idxs(lat, lon)
    a_equal(ret_uids, [22,23,25,26])


    '''
    for (alpha,beta), ij in zip(cube.alpha_betas, cube.gq_indices):
        panel, ei, ej, gi, gj = ij

        if (gi < ngq) and (gj < ngq):
            lat, lon = abp2latlon(alpha+td, beta+td, panel)
            ret_ij = cube.get_surround_ij(lat, lon)
            a_equal(ij, ret_ij)
    '''




def plot_cs_voronoi_scipy():
    '''
    CubeGridRemap.get_voronoi(): Plot with voronoi_plot_2d
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




def plot_cs_voronoi_polygon():
    '''
    CubeGridRemap.get_voronoi(): Plot with cube_basemap.py
    '''
    from time import sleep
    from cube_remap import CubeGridRemap
    from util.convert_coord.cs_ll import xyp2latlon, xyz2latlon
    from util.plot.cube_basemap import PlotSphere, draw_points, draw_polygon

    ne, ngq = 3, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)
    print 'ne=%d, ngq=%d'%(ne, ngq)

    # plot
    ps = PlotSphere(0, 0, figsize=(15,15), interact=True, draw_map=True)
    ps.draw_cube_panel(1)
    ps.draw_cube_panel(4)
    ps.draw_cube_panel(5)
    ps.draw_cube_elements(ne, 1)

    polys = list()
    for uid in xrange(1):
        lat0, lon0 = cube.latlons[uid]

        #xy_vts, vor_obj = cube.get_voronoi_scipy(uid)
        #ll_vts = [xyp2latlon(x,y,1,lat0,lon0) for x,y in xy_vts]

        xyz_vts = cube.get_voronoi(uid)
        ll_vts = [xyz2latlon(*xyz) for xyz in xyz_vts]
        print ll_vts

        draw_points(ps.bmap, ll_vts)
        poly = draw_polygon(ps.bmap, ll_vts)
        poly.update( dict(fc='r') )

        polys.append(poly)
        #ps.draw()

        #polys[uid].set_facecolor('w')

    ps.show(True)




def test_cs_voronoi_area():
    '''
    CubeGridRemap.get_voronoi(): check the sphere area, ne=3
    '''
    from cube_remap import CubeGridRemap
    from util.geometry.sphere import area_polygon
    from math import fsum, pi
    from util.convert_coord.cart_cs import xyp2xyz

    ne, ngq = 3, 4
    rotated = False
    cube = CubeGridRemap(ne, ngq, rotated)

    area_list = list()
    for uid in xrange(cube.up_size):
        #xy_vts, vor_obj = cube.get_voronoi_scipy(uid)
        #xyzs = [xyp2xyz(x,y,1) for x,y in xy_vts]
        #area_list.append( area_polygon(xyzs) )

        xyzs = cube.get_voronoi(uid)
        area_list.append( area_polygon(xyzs) )

    aa_equal(fsum(area_list), 4*pi, 15)

    '''
    # check time (in bricks)
    # ne30  digit=15  149 s
    # ne60  digit=15  594 s
    # ne120 digit=15 2372 s

    for digit in xrange(16,0,-1):
        try:
            aa_equal(fsum(area_list), 4*pi, digit)
            print 'digit=%d'%digit
            break
        except:
            pass
    '''




def test_ll_voronoi_area():
    '''
    LatlonGridRemap.get_voronoi(): check the sphere area
    '''
    from cube_remap import LatlonGridRemap
    from util.geometry.sphere import area_polygon
    from math import fsum, pi
    from util.convert_coord.cart_ll import latlon2xyz

    nlat, nlon = 90, 180
    #nlat, nlon = 180, 360
    #nlat, nlon = 360, 720
    #nlat, nlon = 720, 1440
    ll_obj = LatlonGridRemap(nlat, nlon)

    area_list = list()
    for idx in xrange(ll_obj.nsize):
        #latlons = ll_obj.get_voronoi(idx)
        #xyzs = [latlon2xyz(*latlon) for latlon in latlons]
        xyzs = ll_obj.get_voronoi(idx)
        area_list.append( area_polygon(xyzs) )

    aa_equal(fsum(area_list), 4*pi, 12)

    '''
    # check time (in bricks)
    #  90x180  digit=12   5 s
    # 180x360  digit=12  17 s
    # 360x720  digit=12  69 s
    # 720x1440 digit=12 274 s

    for digit in xrange(16,0,-1):
        try:
            aa_equal(fsum(area_list), 4*pi, digit)
            break
        except:
            pass

    print 'digit=%d'%digit
    '''




def plot_ll_voronoi_polygon():
    '''
    LatlonGridRemap.get_voronoi(): Plot with cube_basemap.py
    '''
    from time import sleep
    from cube_remap import LatlonGridRemap
    from util.convert_coord.cs_ll import xyp2latlon, xyz2latlon
    from util.plot.cube_basemap import PlotSphere, draw_points, draw_polygon

    nlat, nlon = 90, 180
    ll_obj = LatlonGridRemap(nlat, nlon)
    print 'nlat=%d, nlon=%d'%(nlat, nlon)

    # plot
    ps = PlotSphere(-90, 0, figsize=(15,15), interact=True, draw_map=True)

    for idx in xrange(nlon,2*nlon+1):
        lat0, lon0 = ll_obj.latlons[idx]
        #ll_vts = ll_obj.get_voronoi(idx)
        ll_vts = [xyz2latlon(*xyz) for xyz in ll_obj.get_voronoi(idx)]

        draw_points(ps.bmap, ll_vts)
        poly = draw_polygon(ps.bmap, ll_vts)
        poly.update( dict(fc='r') )

    ps.show(True)




def test_get_surround_idxs_latlon():
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
    from util.convert_coord.cart_ll import latlon2xyz

    nlat, nlon = 180, 360
    ll = LatlonGridRemap(nlat, nlon, 'regular')

    ret = ll.get_voronoi(1)
    expect = [(-1.5707963267948966, 0                   ), \
              (-1.5447610285607269, 0.026179938779914945), \
              (-1.5447610285607269, 0.008726646259971647)]
    expect_xyz = [latlon2xyz(*latlon) for latlon in expect]
    aa_equal(expect_xyz, ret, 10)

    ret = ll.get_voronoi(nlon)
    expect = [(-1.5447610285607269,-0.00872664625997164), \
              (-1.5447610285607269, 0.008726646259971647), \
              (-1.5274041630712807, 0.008726646259971647), \
              (-1.5274041630712807,-0.00872664625997164)]
    expect_xyz = [latlon2xyz(*latlon) for latlon in expect]
    aa_equal(expect_xyz, ret, 10)
