#!/usr/bin/env python
#------------------------------------------------------------------------------
# filename  : compare_netcdf.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.28     start
#
#
# Description: 
#   Compare two NetCDF files
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from util.misc.standard_errors import sem_1_2_inf




def print_standard_errors(var1, var2):
    L1, L2, Linf = sem_1_2_inf(var1, var2)
    print '\tL1= %e'%L1
    print '\tL2= %e'%L2
    print '\tLinf= %e'%Linf




def compare_variables(ncf_ref, ncf): 
    for vname in ncf_ref.variables.keys():
        print vname

        if ncf.variables.has_key(vname):
            var_ref = ncf_ref.variables[vname][:] 
            var = ncf.variables[vname][:] 
            print_standard_errors(var_ref, var)
        else:
            pass
            #print 'There is no variable in second file'




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fpath_ref', type=str, help='reference netcdf file path')
    parser.add_argument('fpath', type=str, help='netcdf file path 2')
    args = parser.parse_args()

    ncf_ref = nc.Dataset(args.fpath_ref, 'r')
    ncf = nc.Dataset(args.fpath, 'r')
    compare_variables(ncf_ref, ncf)
