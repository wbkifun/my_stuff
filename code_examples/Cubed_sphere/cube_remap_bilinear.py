#------------------------------------------------------------------------------
# filename  : cube_remap_bilinear.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.16    start
#
#
# Description: 
#   Remap between cubed-sphere and latlon grid
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal




class Bilinear(object):
    def __init__(self, cs_obj, ll_obj):
        self.cs_obj = cs_obj
        self.ll_obj = ll_obj



    def make_dsw_dict_ll2cs(self):
        cs_obj = self.cs_obj
        ll_obj = self.ll_obj

        dsw_dict = dict()       # {dst:[(s,w),(s,w),...],...}
        
        for dst, (lat,lon) in enumerate(cs_obj.latlons):
            idx1, idx2, idx3, idx4 = ll_obj.get_surround_idxs(lat, lon)

            if -1 in [idx1, idx2, idx3, idx4]:
                dsw_dict[dst] = [(idx1, 1.0)]

            else:
                dlat, dlon = ll_obj.dlat, ll_obj.dlon
                lat1, lon1 = ll_obj.latlons[idx1]
                lat2 = lat1 + dlat
                lon2 = lon1 + dlon

                w1 = (lon2-lon)*(lat2-lat)/(dlon*dlat)
                w2 = (lon-lon1)*(lat2-lat)/(dlon*dlat)
                w3 = (lon2-lon)*(lat-lat1)/(dlon*dlat)
                w4 = (lon-lon1)*(lat-lat1)/(dlon*dlat)

                dsw_dict[dst] = [(idx1,w1),(idx2,w2),(idx3,w3),(idx4,w4)]


        return dsw_dict
