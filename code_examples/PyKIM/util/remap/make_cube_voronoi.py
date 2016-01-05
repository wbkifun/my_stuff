#------------------------------------------------------------------------------
# filename  : make_cube_voronoi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.21    start
#             2015.12.23    change voronois.shape 1D -> 2D
#------------------------------------------------------------------------------

from  __future__ import division
import numpy as np
import os
import sys
import netCDF4 as nc
import argparse

from cube_remap import CubeGridRemap
from util.geometry.sphere import area_polygon



def make_netcdf_cube_voronoi(cs_obj, fpath):
    up_size = cs_obj.up_size
    corner_max_size = 6     # max number of voronoi vertices

    center_xyzs = np.zeros((up_size,3), 'f8')
    corner_xyzs = np.zeros((up_size,corner_max_size,3), 'f8')
    corner_sizes = np.zeros(up_size, 'i4')
    corner_areas = np.zeros(up_size, 'f8')

    for dst in xrange(up_size):
        center_xyzs[dst,:] = cs_obj.xyzs[dst]

        voronoi_xyzs = cs_obj.get_voronoi(dst)
        for i, corner_xyz in enumerate(voronoi_xyzs):
            corner_xyzs[dst,i,:] = corner_xyz

        corner_sizes[dst] = len(voronoi_xyzs)
        corner_areas[dst] = area_polygon(voronoi_xyzs)


    #------------------------------------------------------------
    # Save as NetCDF
    #------------------------------------------------------------
    ncf = nc.Dataset(fpath, 'w', format='NETCDF3_CLASSIC') # for pnetcdf
    ncf.description = 'Voronoi diagram on the Cubed-sphere'
    ncf.coordinates = 'cartesian'

    ncf.rotated = str(cs_obj.rotated).lower()
    ncf.ne = cs_obj.ne
    ncf.ngq = cs_obj.ngq
    ncf.ep_size = cs_obj.ep_size
    ncf.up_size = up_size

    ncf.createDimension('up_size', up_size)
    ncf.createDimension('corner_max_size', corner_max_size)
    ncf.createDimension('3', 3)

    vcenter_xyzs = ncf.createVariable('center_xyzs', 'f8', ('up_size','3'))
    vcorner_xyzs = ncf.createVariable('corner_xyzs', 'f8', ('up_size','corner_max_size','3'))
    vcorner_sizes = ncf.createVariable('corner_sizes', 'i4', ('up_size',))
    vcorner_areas = ncf.createVariable('corner_areas', 'f8', ('up_size',))

    vcenter_xyzs[:] = center_xyzs[:]
    vcorner_xyzs[:] = corner_xyzs[:]
    vcorner_sizes[:] = corner_sizes[:]
    vcorner_areas[:] = corner_areas[:]

    ncf.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rotated', action='store_true', help='Korea centered rotation')
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('output_dir', nargs='?', type=str, \
            help='output directory', default='./voronoi/')
    args = parser.parse_args()

    rotated = args.rotated
    ne, ngq = args.ne, 4
    cs_type = 'rotated' if rotated else 'regular'
    output_dir = args.output_dir
    output_fname = 'voronoi_ne%d_%s.nc'%(ne, cs_type)
    output_fpath = output_dir + output_fname


    print 'Generate the Voronoi diagram of cubed-sphere'
    print 'cs_type: %s'%(cs_type)
    print 'ne=%d, ngq=%d'%(ne, ngq)
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


    cs_obj = CubeGridRemap(ne, ngq, rotated)
    make_netcdf_cube_voronoi(cs_obj, output_fpath)
