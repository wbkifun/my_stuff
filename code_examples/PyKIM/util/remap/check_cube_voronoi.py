#------------------------------------------------------------------------------
# filename  : check_cube_voronoi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.21    start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def check_voronoi_area(ncf):
    from math import fsum, pi
    from util.misc.compare_float import feq

    voronoi_areas = ncf.variables['voronoi_areas'][:]
    area_sum = fsum(voronoi_areas)

    for digit in xrange(15,0,-1):
        if feq(area_sum, 4*pi, digit):      # area on sphere
            break

    print 'check area sum: digit %d'%digit




def plot_voronoi(ncf):
    from util.convert_coord.cs_ll import xyp2latlon, xyz2latlon
    from util.plot.cube_basemap import PlotSphere, draw_points, draw_polygon


    # Read a NetCDF file
    gpt_xyzs = ncf.variables['gridpoints'][:]
    voronoi_xyzs = ncf.variables['voronois'][:]
    address = ncf.variables['voronoi_address'][:]


    # Plot with basemap
    ps = PlotSphere(0, 0, figsize=(15,15), interact=True, draw_map=True)
    ps.draw_cube_panel(1)
    ps.draw_cube_panel(4)
    ps.draw_cube_panel(5)
    ps.draw_cube_elements(ne, 1)

    polys = list()
    for uid in xrange(80):
    #for uid in xrange(24,25):
        #print uid
        ll_gpt = xyz2latlon(*gpt_xyzs[uid])
        draw_points(ps.bmap, [ll_gpt], s=30)

        start = address[uid]
        end = address[uid+1] if uid < ncf.up_size-1 else ncf.up_size
        xyz_vts = voronoi_xyzs[start:end,:]
        ll_vts = [xyz2latlon(*xyz) for xyz in xyz_vts]

        draw_points(ps.bmap, ll_vts, c='r')
        poly = draw_polygon(ps.bmap, ll_vts)
        poly.update( dict(fc='r') )

        polys.append(poly)
        #ps.draw()

        #polys[uid].set_facecolor('w')

    ps.show(True)




if __name__ == '__main__':
    import argparse
    import netCDF4 as nc
    from parse import parse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filepath', type=str, help='file path of Voronoi diagram of cubed-sphere')
    args = parser.parse_args()

    fname = args.filepath.split('/')[-1]
    r = parse("voronoi_ne{ne}_{cs_type}.nc", fname)
    ne = int(r['ne'])
    cs_type = r['cs_type']

    ncf = nc.Dataset(args.filepath, 'r', 'NETCDF3_CLASSIC')

    check_voronoi_area(ncf)
    plot_voronoi(ncf)
