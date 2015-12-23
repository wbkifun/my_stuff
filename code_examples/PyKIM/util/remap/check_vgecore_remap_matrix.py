#------------------------------------------------------------------------------
# filename  : check_vgecore_remap_matrix.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.19    start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def check_area_ratio(ncf, SCRIP=False):
    from util.misc.compare_float import feq

    num_links = len( ncf.dimensions['num_links'] )
    dsts = ncf.variables['dst_address'][:]
    srcs = ncf.variables['src_address'][:]
    wgts = ncf.variables['remap_matrix'][:]

    if SCRIP:
        up_size = len( ncf.dimensions['dst_grid_size'] )
        f = np.zeros(up_size, 'f8')
        for i in xrange(num_links):
            dst, src, wgt = dsts[i]-1, srcs[i]-1, wgts[i,0] 
            f[dst] += wgt
            #print i, dst, src, wgt
    else:
        up_size = ncf.up_size
        f = np.zeros(up_size, 'f8')
        for i in xrange(num_links):
            dst, src, wgt = dsts[i], srcs[i], wgts[i]
            f[dst] += wgt
            #print i, dst, src, wgt

    f_digits = np.ones(up_size, 'i4')*(-1)
    num_digits = np.zeros(16, 'i4')
    for i in xrange(up_size):
        for digit in xrange(15,0,-1):
            if feq(f[i], 1, digit):
                f_digits[i] = digit
                num_digits[digit] += 1
                break

        if f_digits[i] == -1:
            f_digits[i] = 0
            num_digits[0] += 1

    for digit in xrange(16):
        print 'digit %d -> %d (%1.2f %%)'%(digit, num_digits[digit], num_digits[digit]/up_size*100)

    equal(sum(num_digits), up_size)




if __name__ == '__main__':
    import argparse
    import netCDF4 as nc
    from parse import parse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filepath', type=str, help='V-GECoRe remap matrix file path')
    args = parser.parse_args()

    fname = args.filepath.split('/')[-1]
    r = parse("remap_{direction}_ne{ne}_{cs_type}_{nlat}x{nlon}_{ll_type}_{method}.nc", fname)
    direction = r['direction']
    ne = int(r['ne'])
    cs_type = r['cs_type']
    nlat, nlon = int(r['nlat']), int(r['nlon'])
    ll_type = r['ll_type']
    method = r['method']

    ncf = nc.Dataset(args.filepath, 'r', 'NETCDF3_CLASSIC')

    check_area_ratio(ncf, SCRIP=False)
