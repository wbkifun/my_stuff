#------------------------------------------------------------------------------
# filename  : test_cube_grid.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.8      start
#             2015.9.11     change to unittest
#
#
# description: 
#   Check the cubed-sphere grid generated by make_cs_grid.py
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal

from cube_grid import CubedSphereGrid




def check_consistency_mvps(mvps):
    for seq, mvp in enumerate(mvps):
        mvp0 = [k for k in mvp if k != -1]

        for m in mvp0:
            mvp1 = [k for k in mvps[m] if k != -1]

            for i in xrange(4):
                mvp1_roll = np.roll(mvp1,i)
                if mvp1_roll[0] == seq: break

            a_equal(mvp0, mvp1_roll)



def check_consistency_uids_gids(is_uvps, uids, gids):
    u_seq = 0
    for seq, is_uvp in enumerate(is_uvps):
        if is_uvp:
            equal(uids[seq], u_seq)
            equal(gids[u_seq], seq)
            u_seq += 1



def check_indices_nbrs(ne, ngq, gq_indices, nbrs):
    ij2seq = dict([(tuple(ij),seq) for seq, ij in enumerate(gq_indices)])

    a_equal(nbrs[ ij2seq[(1,1,1,1,1)] ], 
            [ij2seq[(1,1,1,2,1)], ij2seq[(1,1,1,2,2)], \
             ij2seq[(1,1,1,1,2)], ij2seq[(4,ne,1,ngq-1,2)], \
             ij2seq[(4,ne,1,ngq-1,1)], ij2seq[(5,1,ne,2,ngq-1)], \
             -1, -1])

    a_equal(nbrs[ ij2seq[(1,1,1,2,1)] ], 
            [ij2seq[(1,1,1,3,1)], ij2seq[(1,1,1,3,2)], \
             ij2seq[(1,1,1,2,2)], ij2seq[(1,1,1,1,2)], \
             ij2seq[(1,1,1,1,1)], ij2seq[(5,1,ne,1,ngq-1)], \
             ij2seq[(5,1,ne,2,ngq-1)], ij2seq[(5,1,ne,3,ngq-1)] ])

    a_equal(nbrs[ ij2seq[(1,1,1,ngq,1)] ], 
            [ij2seq[(1,1,1,ngq,2)], ij2seq[(1,1,1,ngq-1,2)], \
             ij2seq[(1,1,1,ngq-1,1)], ij2seq[(5,1,ne,ngq-1,ngq-1)], \
             ij2seq[(5,1,ne,ngq,ngq-1)], ij2seq[(5,2,ne,2,ngq-1)], \
             ij2seq[(5,2,ne,2,ngq)], ij2seq[(1,2,1,2,2)] ])

    a_equal(nbrs[ ij2seq[(1,1,1,1,2)] ], 
            [ij2seq[(1,1,1,1,1)], ij2seq[(1,1,1,2,1)], \
             ij2seq[(1,1,1,2,2)], ij2seq[(1,1,1,2,3)], \
             ij2seq[(1,1,1,1,3)], ij2seq[(4,ne,1,ngq-1,3)], \
             ij2seq[(4,ne,1,ngq-1,2)], ij2seq[(4,ne,1,ngq-1,1)] ])

    a_equal(nbrs[ ij2seq[(1,1,1,2,2)] ], 
            [ij2seq[(1,1,1,1,1)], ij2seq[(1,1,1,2,1)], \
             ij2seq[(1,1,1,3,1)], ij2seq[(1,1,1,3,2)], \
             ij2seq[(1,1,1,3,3)], ij2seq[(1,1,1,2,3)], \
             ij2seq[(1,1,1,1,3)], ij2seq[(1,1,1,1,2)] ])

    a_equal(nbrs[ ij2seq[(1,1,1,ngq,2)] ], 
            [ij2seq[(1,1,1,ngq,3)], ij2seq[(1,1,1,ngq-1,3)], \
             ij2seq[(1,1,1,ngq-1,2)], ij2seq[(1,1,1,ngq-1,1)], \
             ij2seq[(1,1,1,ngq,1)], ij2seq[(1,2,1,2,1)], \
             ij2seq[(1,2,1,2,2)], ij2seq[(1,2,1,2,3)] ])

    a_equal(nbrs[ ij2seq[(1,1,1,1,ngq)] ], 
            [ij2seq[(1,1,1,1,ngq-1)], ij2seq[(1,1,1,2,ngq-1)], \
             ij2seq[(1,1,1,2,ngq)], ij2seq[(1,1,2,2,2)], \
             ij2seq[(1,1,2,1,2)], ij2seq[(4,ne,2,ngq-1,2)], \
             ij2seq[(4,ne,2,ngq-1,1)], ij2seq[(4,ne,1,ngq-1,ngq-1)] ])

    a_equal(nbrs[ ij2seq[(1,1,1,ngq,ngq)] ], 
            [ij2seq[(1,1,1,ngq-1,ngq)], ij2seq[(1,1,1,ngq-1,ngq-1)], \
             ij2seq[(1,1,1,ngq,ngq-1)], ij2seq[(1,2,1,2,ngq-1)], \
             ij2seq[(1,2,1,2,ngq)], ij2seq[(1,2,2,2,2)], \
             ij2seq[(1,2,2,1,2)], ij2seq[(1,1,2,ngq-1,2)] ])




def test_consistency_mvps_3_4():
    '''
    CubedSphereGrid: check consistency of mvps (ne=3,ngq=4)
    '''
    ne, ngq = 3, 4
    csgrid = CubedSphereGrid(ne, ngq)
    check_consistency_mvps(csgrid.mvps)



def test_consistency_uids_gids_3_4():
    '''
    CubedSphereGrid: check consistency of is_uvps, uids and gids (ne=3,ngq=4)
    '''
    ne, ngq = 3, 4
    csgrid = CubedSphereGrid(ne, ngq)
    check_consistency_uids_gids(csgrid.is_uvps, csgrid.uids, csgrid.gids)



def test_indices_nbrs_3_4():
    '''
    CubedSphereGrid: check indices of nbrs (ne=3,ngq=4)
    '''
    ne, ngq = 3, 4
    csgrid = CubedSphereGrid(ne, ngq)
    check_indices_nbrs(ne, ngq, csgrid.gq_indices, csgrid.nbrs)




if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cs_grid_file', type=str, help='cubed-sphere grid NetCDF file')
    args = parser.parse_args()
    fpath = args.cs_grid_file

    ne = int( re.search('ne([0-9]+)',fpath).group(1) )
    ngq = int( re.search('ngq([0-9]+)',fpath).group(1) )
    print 'ne=%d, ngq=%d'%(ne, ngq) 


    ncf = nc.Dataset(fpath, 'r', format='NETCDF4')
    mvps = ncf.variables['mvps'][:]
    is_uvps = ncf.variables['is_uvps'][:]
    uids = ncf.variables['uids'][:]
    gids = ncf.variables['gids'][:]
    gq_indices = ncf.variables['gq_indices'][:]
    nbrs = ncf.variables['nbrs'][:]

    print 'check consistency: mvps...'
    check_consistency_mvps(mvps)

    print 'check consistency: is_uvps, uids and gids...'
    check_consistency_uids_gids(is_uvps, uids, gids)

    print 'check indices: nbrs...'
    check_indices_nbrs(ne, ngq, gq_indices, nbrs)
