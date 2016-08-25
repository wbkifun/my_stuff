#------------------------------------------------------------------------------
# filename  : make_cube_voronoi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.21    start
#             2015.12.23    change voronois.shape 1D -> 2D
#             2016.4.15     convert to Python3
#------------------------------------------------------------------------------

import numpy as np
import os
import sys
import netCDF4 as nc
import argparse

from cube_remap_matrix import CubeGridRemap
from util.geometry.sphere import area_polygon
from util.convert_coord.cart_ll import xyz2latlon



def make_netcdf_cube_voronoi(cs_obj, fpath):
    up_size = cs_obj.up_size
    corner_max_size = 6     # max number of voronoi vertices

    center_xyzs = np.zeros((up_size,3), 'f8')
    center_latlons = np.zeros((up_size,2), 'f8')
    corner_xyzs = np.zeros((up_size,corner_max_size,3), 'f8')
    corner_latlons = np.zeros((up_size,corner_max_size,2), 'f8')
    corner_sizes = np.zeros(up_size, 'i4')
    corner_areas = np.zeros(up_size, 'f8')

    for dst in range(up_size):
        center_xyzs[dst,:] = cs_obj.xyzs[dst]
        center_latlons[dst,:] = cs_obj.latlons[dst]

        voronoi_xyzs = cs_obj.get_voronoi(dst)
        for i, corner_xyz in enumerate(voronoi_xyzs):
            corner_xyzs[dst,i,:] = corner_xyz
            corner_latlons[dst,i,:] = xyz2latlon(*corner_xyz)

        corner_sizes[dst] = len(voronoi_xyzs)
        corner_areas[dst] = area_polygon(voronoi_xyzs)


    #------------------------------------------------------------
    # Save as NetCDF
    #------------------------------------------------------------
    ncf = nc.Dataset(fpath, 'w', format='NETCDF3_64BIT') # for pnetcdf
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
    ncf.createDimension('2', 2)

    vcenter_xyzs = ncf.createVariable('center_xyzs', 'f8', ('up_size','3'))
    vcenter_lls = ncf.createVariable('center_latlons', 'f8', ('up_size','2'))
    vcorner_xyzs = ncf.createVariable('corner_xyzs', 'f8', ('up_size','corner_max_size','3'))
    vcorner_lls = ncf.createVariable('corner_latlons', 'f8', ('up_size','corner_max_size','2'))
    vcorner_sizes = ncf.createVariable('corner_sizes', 'i4', ('up_size',))
    vcorner_areas = ncf.createVariable('corner_areas', 'f8', ('up_size',))

    vcenter_xyzs[:] = center_xyzs[:]
    vcenter_lls[:] = center_latlons[:]
    vcorner_xyzs[:] = corner_xyzs[:]
    vcorner_lls[:] = corner_latlons[:]
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
    output_fname = "voronoi_ne{}_{}.nc".format(ne, cs_type)
    output_fpath = output_dir + output_fname


    print("Generate the Voronoi diagram of cubed-sphere")
    print("cs_type: {}".format(cs_type))
    print("ne={}, ngq={}".format(ne, ngq))
    print("output directory: {}".format(output_dir))
    print("output filename: {}".format(output_fname))
    
    yn = input("Continue (Y/n)? ")
    if yn.lower() == "n":
        sys.exit()

    if not os.path.exists(output_dir):
        print("{} is not found. Make output directory.".format(output_dir))
        os.makedirs(output_dir)

    if os.path.exists(output_fpath):
        yn = raw_input("{} is found. Overwrite(Y/n)? ".format(output_fpath))
        if yn.lower() == 'n':
            sys.exit()


    cs_obj = CubeGridRemap(ne, ngq, rotated)
    make_netcdf_cube_voronoi(cs_obj, output_fpath)
