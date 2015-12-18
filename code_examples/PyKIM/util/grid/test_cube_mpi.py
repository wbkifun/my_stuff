#------------------------------------------------------------------------------
# filename  : test_cube_mpi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.11     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from math import fsum
import subprocess as subp
import netCDF4 as nc

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup

from util.misc.compare_float import feq
from cube_mpi import CubeGridMPI, CubeMPI
from path import dir_cs_grid




def pre_send(cubempi, f, recv_buf, send_buf):
    dsts = cubempi.send_dsts
    srcs = cubempi.send_srcs
    wgts = cubempi.send_wgts
    local_src_size = cubempi.local_src_size

    recv_buf[:] = 0
    send_buf[:] = 0

    for seq, (dst, src, wgt) in enumerate(zip(dsts,srcs,wgts)):
        if seq < local_src_size:
            recv_buf[dst] += f[src]*wgt
        else:
            send_buf[dst] += f[src]*wgt




def post_recv(cubempi, f, recv_buf):
    dsts = cubempi.recv_dsts
    srcs = cubempi.recv_srcs

    prev_dst = -1
    for dst, src in zip(dsts, srcs):
        if prev_dst != dst: 
            f[dst] = 0
            prev_dst = dst

        f[dst] += recv_buf[src]




def test_avg_random():
    '''
    CubeMPI for AVG: Random values (ne=5, ngq=4, nproc=1)
    '''
    ne, ngq = 5, 4

    nproc, myrank = 1, 0
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, method='AVG')

    a_equal(cubegrid.local_gids, np.arange(6*ne*ne*ngq*ngq))
    a_equal(cubempi.recv_schedule.shape, (0,3))
    a_equal(cubempi.send_schedule.shape, (0,3))
    a_equal(cubempi.recv_buf_size, 6*ne*ne*12)
    a_equal(cubempi.send_buf_size, 0)


    #-----------------------------------------------------
    # Generate a random field on the cubed-sphere
    #-----------------------------------------------------
    ep_size = cubegrid.ep_size

    f = np.random.rand(ep_size)


    #-----------------------------------------------------
    # Average the element boundary for the spectral element method
    #-----------------------------------------------------
    recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')
    send_buf = np.zeros(cubempi.send_buf_size, 'f8')

    pre_send(cubempi, f, recv_buf, send_buf)
    post_recv(cubempi, f, recv_buf)


    #-----------------------------------------------------
    # Check if mvps have same values
    #-----------------------------------------------------
    cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
    cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
    mvps = cs_ncf.variables['mvps'][:]

    for seq, mvp in enumerate(mvps):
        eff_mvp = [k for k in mvp if k != -1]

        for m in eff_mvp:
            aa_equal(f[seq], f[m], 15)




def test_avg_sequential_3_4_1():
    '''
    CubeMPI for AVG: Exact squential values (ne=3, ngq=4, nproc=1)
    '''
    ne, ngq = 3, 4

    nproc, myrank = 1, 0
    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, method='AVG')

    a_equal(cubegrid.local_gids, np.arange(6*ne*ne*ngq*ngq))
    a_equal(cubempi.recv_schedule.shape, (0,3))
    a_equal(cubempi.send_schedule.shape, (0,3))
    a_equal(cubempi.recv_buf_size, 6*ne*ne*12)
    a_equal(cubempi.send_buf_size, 0)


    #-----------------------------------------------------
    # Generate a sequential field on the cubed-sphere
    #-----------------------------------------------------
    f = np.arange(cubegrid.local_ep_size, dtype='f8')


    #-----------------------------------------------------
    # Average the element boundary for the spectral element method
    #-----------------------------------------------------
    recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')
    send_buf = np.zeros(cubempi.send_buf_size, 'f8')

    pre_send(cubempi, f, recv_buf, send_buf)
    post_recv(cubempi, f, recv_buf)


    #-----------------------------------------------------
    # Check if mvps have same values
    #-----------------------------------------------------
    fs = [f]
    ranks, lids = cubegrid.ranks, cubegrid.lids

    cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
    cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
    mvps = cs_ncf.variables['mvps'][:]
    for seq, mvp in enumerate(mvps):
        eff_mvp = [k for k in mvp if k != -1]

        for gid in eff_mvp:
            rank, lid = ranks[gid], lids[gid]
            ok_( feq(fs[rank][lid], np.mean(eff_mvp), 15) )




def test_avg_sequential_3_4_2():
    '''
    CubeMPI for AVG: Exact squential values (ne=3, ngq=4, nproc=2)
    '''
    ne, ngq = 3, 4

    nproc = 2
    cubegrid0 = CubeGridMPI(ne, ngq, nproc, 0)
    cubempi0 = CubeMPI(cubegrid0, method='AVG')

    cubegrid1 = CubeGridMPI(ne, ngq, nproc, 1)
    cubempi1 = CubeMPI(cubegrid1, method='AVG')


    # Check send/recv pair in send_group and recv_group
    a_equal(cubempi0.send_group[1].keys(), cubempi1.recv_group[0])
    a_equal(cubempi0.recv_group[1], cubempi1.send_group[0].keys())


    # Check send/recv pair in send_buf and recv_buf
    rank0, start0, size0 = cubempi0.send_schedule[0]
    rank1, start1, size1 = cubempi1.recv_schedule[0]
    i0, i1 = start0, start0+size0
    i2, i3 = start1, start1+size1
    a_equal(cubempi0.send_buf[i0:i1], cubempi1.recv_buf[i2:i3])

    rank0, start0, size0 = cubempi0.recv_schedule[0]
    rank1, start1, size1 = cubempi1.send_schedule[0]
    i0, i1 = start0, start0+size0
    i2, i3 = start1, start1+size1
    a_equal(cubempi0.recv_buf[i0:i1], cubempi1.send_buf[i2:i3])


    #-----------------------------------------------------
    # Generate a sequential field on the cubed-sphere
    #-----------------------------------------------------
    f0 = cubegrid0.local_gids.astype('f8')
    f1 = cubegrid1.local_gids.astype('f8')


    #-----------------------------------------------------
    # Average the element boundary for the spectral element method
    #-----------------------------------------------------
    recv_buf0 = np.zeros(cubempi0.recv_buf_size, 'f8')
    send_buf0 = np.zeros(cubempi0.send_buf_size, 'f8')
    recv_buf1 = np.zeros(cubempi1.recv_buf_size, 'f8')
    send_buf1 = np.zeros(cubempi1.send_buf_size, 'f8')

    # Prepare to send
    pre_send(cubempi0, f0, recv_buf0, send_buf0)
    pre_send(cubempi1, f1, recv_buf1, send_buf1)

    # Send/Recv
    rank0, start0, size0 = cubempi0.send_schedule[0]
    rank1, start1, size1 = cubempi1.recv_schedule[0]
    recv_buf1[start1:start1+size1] = send_buf0[start0:start0+size0]

    rank1, start1, size1 = cubempi1.send_schedule[0]
    rank0, start0, size0 = cubempi0.recv_schedule[0]
    recv_buf0[start0:start0+size0] = send_buf1[start1:start1+size1]

    # After receive
    post_recv(cubempi0, f0, recv_buf0)
    post_recv(cubempi1, f1, recv_buf1)


    #-----------------------------------------------------
    # Check if mvps have same values
    #-----------------------------------------------------
    fs = [f0, f1]
    ranks, lids = cubegrid0.ranks, cubegrid0.lids

    cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
    cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
    mvps = cs_ncf.variables['mvps'][:]
    for seq, mvp in enumerate(mvps):
        eff_mvp = [k for k in mvp if k != -1]

        for gid in eff_mvp:
            rank, lid = ranks[gid], lids[gid]
            ok_( feq(fs[rank][lid], np.mean(eff_mvp), 15) )




def test_avg_sequential_3_4_3():
    '''
    CubeMPI for AVG: Exact squential values (ne=3, ngq=4, nproc=3)
    '''
    ne, ngq = 3, 4

    nproc = 3
    cubegrid0 = CubeGridMPI(ne, ngq, nproc, 0)
    cubempi0 = CubeMPI(cubegrid0, method='AVG')

    cubegrid1 = CubeGridMPI(ne, ngq, nproc, 1)
    cubempi1 = CubeMPI(cubegrid1, method='AVG')
    
    cubegrid2 = CubeGridMPI(ne, ngq, nproc, 2)
    cubempi2 = CubeMPI(cubegrid2, method='AVG')

    # Check send/recv pair in send_group and recv_group
    a_equal(cubempi0.send_group[1].keys(), cubempi1.recv_group[0])
    a_equal(cubempi0.send_group[2].keys(), cubempi2.recv_group[0])
    a_equal(cubempi1.send_group[0].keys(), cubempi0.recv_group[1])
    a_equal(cubempi1.send_group[2].keys(), cubempi2.recv_group[1])
    a_equal(cubempi2.send_group[0].keys(), cubempi0.recv_group[2])
    a_equal(cubempi2.send_group[1].keys(), cubempi1.recv_group[2])

    # Check send/recv pair in send_buf and recv_buf
    rank0, i0, n0 = cubempi0.send_schedule[0]   # send 0->1
    rank1, i1, n1 = cubempi1.recv_schedule[0]   # recv 
    a_equal(cubempi0.send_buf[i0:i0+n0], cubempi1.recv_buf[i1:i1+n1])

    rank0, i0, n0 = cubempi1.send_schedule[0]   # send 1->0
    rank1, i1, n1 = cubempi0.recv_schedule[0]   # recv
    a_equal(cubempi1.send_buf[i0:i0+n0], cubempi0.recv_buf[i1:i1+n1])

    rank0, i0, n0 = cubempi0.send_schedule[1]   # send 0->2
    rank1, i1, n1 = cubempi2.recv_schedule[0]   # recv
    a_equal(cubempi0.send_buf[i0:i0+n0], cubempi2.recv_buf[i1:i1+n1])

    rank0, i0, n0 = cubempi2.send_schedule[0]   # send 2->0
    rank1, i1, n1 = cubempi0.recv_schedule[1]   # recv
    a_equal(cubempi2.send_buf[i0:i0+n0], cubempi0.recv_buf[i1:i1+n1])

    rank0, i0, n0 = cubempi1.recv_schedule[1]   # send 1->2 
    rank1, i1, n1 = cubempi2.send_schedule[1]   # recv
    a_equal(cubempi1.recv_buf[i0:i0+n0], cubempi2.send_buf[i1:i1+n1])

    rank0, i0, n0 = cubempi2.recv_schedule[1]   # send 2->1 
    rank1, i1, n1 = cubempi1.send_schedule[1]   # recv
    a_equal(cubempi2.recv_buf[i0:i0+n0], cubempi1.send_buf[i1:i1+n1])


    #-----------------------------------------------------
    # Generate a sequential field on the cubed-sphere
    #-----------------------------------------------------
    f0 = cubegrid0.local_gids.astype('f8')
    f1 = cubegrid1.local_gids.astype('f8')
    f2 = cubegrid2.local_gids.astype('f8')


    #-----------------------------------------------------
    # Average the element boundary for the spectral element method
    #-----------------------------------------------------
    recv_buf0 = np.zeros(cubempi0.recv_buf_size, 'f8')
    send_buf0 = np.zeros(cubempi0.send_buf_size, 'f8')
    recv_buf1 = np.zeros(cubempi1.recv_buf_size, 'f8')
    send_buf1 = np.zeros(cubempi1.send_buf_size, 'f8')
    recv_buf2 = np.zeros(cubempi2.recv_buf_size, 'f8')
    send_buf2 = np.zeros(cubempi2.send_buf_size, 'f8')

    # Prepare to send
    pre_send(cubempi0, f0, recv_buf0, send_buf0)
    pre_send(cubempi1, f1, recv_buf1, send_buf1)
    pre_send(cubempi2, f2, recv_buf2, send_buf2)

    # Send/Recv
    rank0, i0, n0 = cubempi0.send_schedule[0]    # send 0->1
    rank1, i1, n1 = cubempi1.recv_schedule[0]
    recv_buf1[i1:i1+n1] = send_buf0[i0:i0+n0]

    rank1, i1, n1 = cubempi1.send_schedule[0]    # send 1->0
    rank0, i0, n0 = cubempi0.recv_schedule[0]
    recv_buf0[i0:i0+n0] = send_buf1[i1:i1+n1]

    rank1, i1, n1 = cubempi0.send_schedule[1]    # send 0->2
    rank0, i0, n0 = cubempi2.recv_schedule[0]
    recv_buf2[i0:i0+n0] = send_buf0[i1:i1+n1]

    rank1, i1, n1 = cubempi2.send_schedule[0]    # send 2->0
    rank0, i0, n0 = cubempi0.recv_schedule[1]
    recv_buf0[i0:i0+n0] = send_buf2[i1:i1+n1]

    rank1, i1, n1 = cubempi1.send_schedule[1]    # send 1->2
    rank0, i0, n0 = cubempi2.recv_schedule[1]
    recv_buf2[i0:i0+n0] = send_buf1[i1:i1+n1]

    rank1, i1, n1 = cubempi2.send_schedule[1]    # send 2->1
    rank0, i0, n0 = cubempi1.recv_schedule[1]
    recv_buf1[i0:i0+n0] = send_buf2[i1:i1+n1]

    # After receive
    post_recv(cubempi0, f0, recv_buf0)
    post_recv(cubempi1, f1, recv_buf1)
    post_recv(cubempi2, f2, recv_buf2)


    #-----------------------------------------------------
    # Check if mvps have same values
    #-----------------------------------------------------
    fs = [f0, f1, f2]
    ranks, lids = cubegrid0.ranks, cubegrid0.lids

    cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
    cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
    mvps = cs_ncf.variables['mvps'][:]
    for seq, mvp in enumerate(mvps):
        eff_mvp = [k for k in mvp if k != -1]

        for gid in eff_mvp:
            rank, lid = ranks[gid], lids[gid]
            ok_( feq(fs[rank][lid], np.mean(eff_mvp), 15) )




def check_avg_sequential_mpi(ne, ngq, comm):
    '''
    CubeMPI for AVG
    '''
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank)
    cubempi = CubeMPI(cubegrid, method='AVG', comm=None)


    # Generate a sequential field on the cubed-sphere
    f = cubegrid.local_gids.astype('f8')

    # Average the element boundary for the spectral element method
    send_buf = np.zeros(cubempi.send_buf_size, 'f8')
    recv_buf = np.zeros(cubempi.recv_buf_size, 'f8')

    # Send/Recv
    pre_send(cubempi, f, recv_buf, send_buf)

    req_send_list = list()
    req_recv_list = list()

    for dest, start, size in cubempi.send_schedule:
        req = comm.Isend(send_buf[start:start+size], dest, 0)
        req_send_list.append(req)

    for dest, start, size in cubempi.recv_schedule:
        req = comm.Irecv(recv_buf[start:start+size], dest, 0)
        req_recv_list.append(req)

    MPI.Request.Waitall(req_send_list)
    MPI.Request.Waitall(req_recv_list)

    # After receive
    post_recv(cubempi, f, recv_buf)


    #-----------------------------------------------------
    # Check if mvps have same values
    #-----------------------------------------------------
    if myrank == 0:
        fs = [f]
        for src in xrange(1,nproc):
            size = cubegrid.local_ep_size
            fs.append( np.zeros(size, 'f8') )
            comm.Recv(fs[-1], src, 10)

        cs_fpath = dir_cs_grid + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
        mvps = cs_ncf.variables['mvps'][:]
        ranks, lids = cubegrid.ranks, cubegrid.lids
        for seq, mvp in enumerate(mvps):
            eff_mvp = [k for k in mvp if k != -1]

            for gid in eff_mvp:
                rank, lid = ranks[gid], lids[gid]
                ok_( feq(fs[rank][lid], np.mean(eff_mvp), 15) )

    else:
        comm.Send(f, 0, 10)




def run_mpi(ne, ngq, nproc):
    cmd = 'mpirun -np %d python %s %d %d'%(nproc, __file__, ne, ngq)
    proc = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = proc.communicate()
    #print stdout
    assert stderr == '', stderr




def test_avg_sequential_mpi():
    ne, ngq = 3, 4
    for nproc in xrange(1,33):
        func = with_setup(lambda:None, lambda:None)(lambda:run_mpi(ne,ngq,nproc)) 
        func.description = 'CubeMPI for AVG: Sequential values with MPI (ne=%d, ngq=%d, nproc=%d)'%(ne,ngq,nproc)
        yield func




if __name__ == '__main__':
    import argparse
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('ngq', type=int, help='number of Gauss qudrature points')
    args = parser.parse_args()
    #if myrank == 0: print 'ne=%d, ngq=%d'%(args.ne, args.ngq) 

    check_avg_sequential_mpi(args.ne, args.ngq, comm)
