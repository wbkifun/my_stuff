#------------------------------------------------------------------------------
# filename  : make_cube_voronoi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.21    start
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
    gridpoints = np.zeros((cs_obj.up_size,3), 'f8')
    voronois = list()
    voronoi_address = np.zeros(cs_obj.up_size, 'i4')
    voronoi_areas = np.zeros(cs_obj.up_size, 'f8')

    seq = 0
    for dst in xrange(cs_obj.up_size):
        gridpoints[dst,:] = cs_obj.xyzs[dst]
        voronoi = cs_obj.get_voronoi(dst)
        voronois.extend(voronoi)
        voronoi_address[dst] = seq
        voronoi_areas[dst] = area_polygon(voronoi)

        seq += len(voronoi)

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
    ncf.up_size = cs_obj.up_size

    ncf.createDimension('up_size', cs_obj.up_size)
    ncf.createDimension('voronoi_size', len(voronois))
    ncf.createDimension('3', 3)

    vgridpoints = ncf.createVariable('gridpoints', 'f8', ('up_size','3'))
    vvoronois = ncf.createVariable('voronois', 'f8', ('voronoi_size','3'))
    vvoronoi_address = ncf.createVariable('voronoi_address', 'i4', ('up_size',))
    vvoronoi_areas = ncf.createVariable('voronoi_areas', 'f8', ('up_size',))

    vgridpoints[:] = gridpoints[:]
    vvoronois[:] = np.array(voronois, 'f8')
    vvoronoi_address[:] = voronoi_address[:]
    vvoronoi_areas[:] = voronoi_areas[:]

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
