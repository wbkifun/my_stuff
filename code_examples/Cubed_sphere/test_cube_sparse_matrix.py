#------------------------------------------------------------------------------
# filename  : test_cube_sparse_matrix.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.9      srart
#
#
# description: 
#   Check the sparse matrix for the spectral element method
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from math import fsum

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

from pkg.util.compare_float import feq

from cube_sparse_matrix import SparseMatrixSE




def check_sparse_matrix_on_mvp(f, dsts, srcs, weights, mvps):
    '''
    Check same values on the MVP with random numbers...
    '''
    #-----------------------------------------------------
    # Average the element boundary for the spectral element method
    # using the given sparse matrix
    #-----------------------------------------------------
    unique_dsts, index_dsts = np.unique(dsts, return_index=True)
    dst_group = list(index_dsts) + [len(dsts)]

    tmp = np.zeros(len(unique_dsts), 'f8')
    for u_seq, (start, end) in enumerate(zip(dst_group[:-1], dst_group[1:])):
        ws_list = [weights[i]*f[srcs[i]] for i in xrange(start,end)]
        tmp[u_seq] = fsum(ws_list)

    for u_seq, u_dst in enumerate(unique_dsts):
        f[u_dst] = tmp[u_seq]


    #-----------------------------------------------------
    # Check if mvps have same values
    #-----------------------------------------------------
    for seq, mvp in enumerate(mvps):
        eff_mvp = [k for k in mvp if k != -1]

        for m in eff_mvp:
            a_equal(f[seq], f[m])




def check_exact_value_mvp(ne, ngq, dsts, srcs, weights, mvps, gq_indices):
    '''
    Check the exact value on MVP
    '''
    f = np.arange(mvps.shape[0], dtype='f8')
    check_sparse_matrix_on_mvp(f, dsts, srcs, weights, mvps)

    ij2seq = dict([(tuple(ij),seq) for seq, ij in enumerate(gq_indices)])


    ret = (ij2seq[(1,1,1,1,1)] + 
           ij2seq[(4,ne,1,ngq,1)] + 
           ij2seq[(5,1,ne,1,ngq)])/3
    ok_( feq(f[ ij2seq[(1,1,1,1,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(4,ne,1,ngq,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(5,1,ne,1,ngq)] ], ret, 15) )


    ret = (ij2seq[(1,1,1,2,1)] + 
           ij2seq[(5,1,ne,2,ngq)])/2
    ok_( feq(f[ ij2seq[(1,1,1,2,1)] ], ret, 15) ) 
    ok_( feq(f[ ij2seq[(5,1,ne,2,ngq)] ], ret, 15) ) 


    ret = (ij2seq[(1,1,1,ngq,1)] + 
           ij2seq[(5,1,ne,ngq,ngq)] + 
           ij2seq[(5,2,ne,1,ngq)] + 
           ij2seq[(1,2,1,1,1)])/4
    ok_( feq(f[ ij2seq[(1,1,1,ngq,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(5,1,ne,ngq,ngq)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(5,2,ne,1,ngq)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(1,2,1,1,1)] ], ret, 15) )


    ret = (ij2seq[(1,1,1,1,2)] + 
           ij2seq[(4,ne,1,ngq,2)])/2
    ok_( feq(f[ ij2seq[(1,1,1,1,2)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(4,ne,1,ngq,2)] ], ret, 15) ) 


    ret = (ij2seq[(2,1,1,ngq,1)] + 
           ij2seq[(5,ne,ne,ngq,1)] + 
           ij2seq[(5,ne,ne-1,ngq,ngq)] + 
           ij2seq[(2,2,1,1,1)])/4
    ok_( feq(f[ ij2seq[(2,1,1,ngq,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(5,ne,ne,ngq,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(5,ne,ne-1,ngq,ngq)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(2,2,1,1,1)] ], ret, 15) )


    ret = (ij2seq[(2,ne,1,ngq,1)] + 
           ij2seq[(5,ne,1,ngq,1)] + 
           ij2seq[(3,1,1,1,1)])/3
    ok_( feq(f[ ij2seq[(2,ne,1,ngq,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(5,ne,1,ngq,1)] ], ret, 15) )
    ok_( feq(f[ ij2seq[(3,1,1,1,1)] ], ret, 15) )




def test_se_random():
    '''
    SparseMatrixSE: Same values on the MVP with random numbers (ne=5,ngq=4)
    '''
    ne, ngq = 5, 4
    spmat = SparseMatrixSE(ne, ngq)

    ep_size = len( spmat.cs_ncf.dimensions['ep_size'] )
    mvps = spmat.cs_ncf.variables['mvps'][:]

    f = np.random.rand(ep_size)
    check_sparse_matrix_on_mvp(f, spmat.dsts, spmat.srcs, spmat.weights, mvps)




def test_se_sequential():
    '''
    SparseMatrixSE: Same values on the MVP with sequential numbers (ne=5,ngq=4)
    '''
    ne, ngq = 5, 4
    spmat = SparseMatrixSE(ne, ngq)

    mvps = spmat.cs_ncf.variables['mvps'][:]
    gq_indices = spmat.cs_ncf.variables['gq_indices'][:]

    check_exact_value_mvp(ne, ngq, spmat.dsts, spmat.srcs, spmat.weights, mvps, gq_indices)




def test_se_sequential_3_4():
    '''
    SparseMatrixSE: Exact values on the MVP with sequential numbers (ne=3,ngq=4)
    '''
    ne, ngq = 3, 4
    spmat = SparseMatrixSE(ne, ngq)
    mvps = spmat.cs_ncf.variables['mvps'][:]

    f = np.arange(mvps.shape[0], dtype='f8')
    check_sparse_matrix_on_mvp(f, spmat.dsts, spmat.srcs, spmat.weights, mvps)

    ret = (0 + 467 + 684)/3 
    ok_( feq(f[0], ret, 15) )
    ok_( feq(f[467], ret, 15) )
    ok_( feq(f[684], ret, 15) )

    ret = (1 + 685)/2
    ok_( feq(f[1], ret, 15) ) 
    ok_( feq(f[685], ret, 15) ) 

    ret = (3 + 687 + 700 + 16)/4
    ok_( feq(f[3], ret, 15) )
    ok_( feq(f[687], ret, 15) )
    ok_( feq(f[700], ret, 15) )
    ok_( feq(f[16], ret, 15) )

    ret = (4 + 471)/2
    ok_( feq(f[4], ret, 15) )
    ok_( feq(f[471], ret, 15) ) 

    ret = (147 + 707 + 671 + 160)/4
    ok_( feq(f[147], ret, 15) )
    ok_( feq(f[707], ret, 15) )
    ok_( feq(f[671], ret, 15) )
    ok_( feq(f[160], ret, 15) )

    ret = (179 + 611 + 288)/3
    ok_( feq(f[179], ret, 15) )
    ok_( feq(f[611], ret, 15) )
    ok_( feq(f[288], ret, 15) )




if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sparse_matrix_file', type=str, help='sparse matrix NetCDF file')
    args = parser.parse_args()
    fpath = args.sparse_matrix_file

    ne = int( re.search('ne([0-9]+)',fpath).group(1) )
    ngq = int( re.search('ngq([0-9]+)',fpath).group(1) )
    print 'ne=%d, ngq=%d'%(ne, ngq) 


    # Read NetCDF files
    cs_fpath = './cs_grid_ne%dngq%d.nc'%(ne,ngq)
    cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
    ep_size = len( cs_ncf.dimensions['ep_size'] )
    mvps = cs_ncf.variables['mvps'][:]
    gq_indices = cs_ncf.variables['gq_indices'][:]

    spmat_ncf = nc.Dataset(fpath, 'r', format='NETCDF4')
    dsts = spmat_ncf.variables['dsts'][:]
    srcs = spmat_ncf.variables['srcs'][:]
    weights = spmat_ncf.variables['weights'][:]


    # Check the sparse matrix
    print 'Check same values on the MVP with random numbers...'
    f = np.random.rand(ep_size)
    check_sparse_matrix_on_mvp(f, dsts, srcs, weights, mvps)

    print 'Check same values on the MVP with sequential numbers...'
    check_exact_value_mvp(ne, ngq, dsts, srcs, weights, mvps, gq_indices)
