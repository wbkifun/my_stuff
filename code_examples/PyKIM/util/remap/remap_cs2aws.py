#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : remap_cs2aws.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.8.4      start
#
#
# Description: 
#   Remapping Cubed-sphere -> AWS Korea
#------------------------------------------------------------------------------

import numpy as np
import netCDF4 as nc
import glob
import os




if __name__ == '__main__':
    aws_fpath = '/data/VERF/mid_verf_work/LIST.DIR/stn.dat'
    remap_matrix_fpath = '/data/KIM2.3/remap_matrix/ne060np4_rotated/vgecore/cs2aws.nc'
    model_dpath = '/data/VERF/KIM/OUTPUT/TBD/v2.4/Medium/ne60np4/201307/'
    out_dpath = './kim_rain_aws/'

    #--------------------------------------------------------------------------
    # Read AWS info
    #--------------------------------------------------------------------------
    with open(aws_fpath, 'r') as f:
        stn_ids = np.array([np.int32(line.split()[0]) for line in f.readlines()])
        print('AWS: nsize {}'.format(len(stn_ids)))
        #print(stn_ids)


    #--------------------------------------------------------------------------
    # Read KIM output files
    # Remapping (ncrain + crain) to AWS locations
    #--------------------------------------------------------------------------
    dpath_list = sorted(glob.glob(model_dpath+'*'))
    for dpath in dpath_list:
        print('\n{}/'.format(dpath))

        for hour in range(6,72+1,6):
            fpath = dpath + '/' + 'sfc.ft{:02d}.nc'.format(hour)
            print(fpath),
            if os.path.exists(fpath):
                print('')
            else:
                print('-> Not found')


            # Read ncrain and crain from the NetCDF file
            ncf = nc.Dataset(fpath, 'r')
            ncrain = ncf.variables['ncrain'][0,:]
            crain = ncf.variables['crain'][0,:]
