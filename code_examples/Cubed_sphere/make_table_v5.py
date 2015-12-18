#------------------------------------------------------------------------------
# filename  : make_table_v5.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.11.13    start
#
# description: 
#   Generate the index tables for MPI parallel on the cubed-sphere
#------------------------------------------------------------------------------

from __future__ import division

from cube_mpi import CubeGridMPI, CubeMPI



class CubeMPITask(CubeMPI):
    def __init__(self, cubegrid, method, comm):
        self.cubegrid = cubegrid
        self.method = method        # method represented by the sparse matrix

        self.ne = cubegrid.ne
        self.ngq = cubegrid.ngq
        self.nproc = cubegrid.nproc
        self.myrank = myrank = cubegrid.myrank
        self.ranks = cubegrid.ranks
        self.lids = cubegrid.lids

        self.arr_dict = dict()




if __name__ == '__main__':
    import os
    import argparse
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myrank = comm.Get_rank()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('target_nproc', type=int, help='number of target MPI processes')
    args = parser.parse_args()

    ngq = 4
    ne = args.ne
    target_nproc = args.target_nproc
    #dpath = '/scratch/khkim/mpi_tables_ne%d_nproc%d'%(ne,target_nproc)
    dpath = './mpi_tables_ne%d_nproc%d'%(ne,target_nproc)


    if myrank == 0:
        print 'Generate the MPI tables for Implicit diffusion'
        print 'ne=%d, ngq=%d, target_nproc=%d'%(ne,ngq,target_nproc)

        if not os.path.exists(dpath):
            os.makedirs(dpath)

        cubegrid = CubeGridMPI(ne, ngq, target_nproc, myrank, homme_style=True)
        cubempi = CubeMPITask(cubegrid, 'IMPVIS', comm)
        cubempi.read_sparse_matrix()

        for target_rank in xrange(target_nproc):
            rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)

            arr_dict = cubempi.extract_local_sparse_matrix(target_rank)
            arr_dict['target_rank'] = target_rank
            comm.send(arr_dict, dest=rank, tag=10)

        for proc in xrange(nproc-1):
            rank = comm.recv(source=MPI.ANY_SOURCE, tag=0)
            comm.send('quit', dest=rank, tag=10)

    else:
        while True:
            comm.send(myrank, dest=0, tag=0)
            msg = comm.recv(source=0, tag=10)

            if msg == 'quit':
                print 'Slave: rank %d quit'%(myrank)
                break

            else:
                target_rank = msg.pop('target_rank')
                print 'Slave: target_rank: %d'%(target_rank)

                cubegrid = CubeGridMPI(ne, ngq, target_nproc, target_rank, homme_style=True)
                cubempi = CubeMPITask(cubegrid, 'IMPVIS', comm)

                cubempi.arr_dict.update(msg)
                cubempi.make_mpi_tables()
                cubempi.save_netcdf(dpath, 'Implicit Viscosity', 'NETCDF3_CLASSIC')
