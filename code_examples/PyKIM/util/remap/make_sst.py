#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : make_sst.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.3.4      start
#
#
# Description: 
#   Generate SST ancillaries for the cubed-sphere grid
#------------------------------------------------------------------------------

from __future__ import division

from util.remap.remap_ancillary import RemapAncillary



date_list = ['201307%.2d00'%(day+1) for day in xrange(31)]
ne_list = [30, 60, 90]
cs_type_list = ['regular', 'rotated']
remap_matrix_dir = '/data/KIM2.3/remap_matrix/'
src_dir = '/data/KIM2.3/ancillary_preprocessed/sst/'
dst_dir = '/data/KIM2.3/inputdata/'

for cs_type in cs_type_list:
    for ne in ne_list:
        ancil = RemapAncillary(ne, cs_type, remap_matrix_dir, src_dir, dst_dir)

        for date in date_list:
            ancil.sea_surface_temperature_gfs(date)
