#------------------------------------------------------------------------------
# filename  : check_bilinear_remap_matrix.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.15     start
#
#
# Description: 
#   Check remap matrix for bilinear if weight is negative 
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import os
import fnmatch

import netCDF4 as nc
from numpy import pi
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from util.grid.path import dir_cs_grid
from util.convert_coord.cs_ll import latlon2abp
from util.convert_coord.cart_ll import latlon2xyz
from util.geometry.voronoi import get_voronoi_scipy, get_voronoi_xyzs




def find_fpaths(directory, pattern):
    for root, dirs, files in os.walk(directory):
        #print root, dirs, files
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename 
                
                

def find_dir_fpaths(directory, dirname):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir == dirname:
                for root2, dirs2, files2 in os.walk( os.path.join(root,dir) ):
                    for basename2 in files2:
                        filename = os.path.join(root2, basename2)
                        yield filename 
                

                
def check_negative(fpath):
    ncf = nc.Dataset(fpath, 'r')
    weights = ncf.variables['remap_matrix'][:]
    '''
    for dst, wgts in enumerate(weights):
        if np.any(wgts<0):
            print wgts
    '''

    return 'Found negative weight' if np.any(weights<0) else 'OK'




if __name__ == '__main__':
    import argparse
    import os
    import sys


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('search_directory', type=str, help='directory path to search remap matrix for bilinear')
    args = parser.parse_args()


    dpath = args.search_directory
    for fpath in find_fpaths(dpath, '*_bilinear.nc'):
        print fpath, '...', check_negative(fpath)

    for fpath in find_dir_fpaths(dpath, 'bilinear'):
        print fpath, '...', check_negative(fpath)
