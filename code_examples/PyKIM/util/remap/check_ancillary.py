#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : check_ancillary.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.2.24     start
#
#
# Description: 
#   Check input files for KIM
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import os
import fnmatch
import netCDF4 as nc




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
                

                
def check_resolution(ncf):
    ncf.rotated == ''
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
    parser.add_argument('search_directory', type=str, help='directory path to search ancillary files')
    args = parser.parse_args()


    dpath = args.search_directory
    for fpath in find_fpaths(dpath, '*.nc'):
        ncf = nc.Dataset(fpath, 'r')
        print fpath, '...', check_negative(fpath)

    #for fpath in find_dir_fpaths(dpath, 'bilinear'):
    #    print fpath, '...', check_negative(fpath)
