from __future__ import division
import numpy
import netCDF4 as nc
import os
        
        

        
class InteractBetweenElems(object):
    def __init__(self, ne, ngq, mpi_comm=None):
        self.ne = ne
        self.ngq = ngq
        self.mpi_comm = mpi_comm

        if mpi_comm == None:
            rank = 1
            nproc = 1
        else:
            rank = mpi_comm.Get_rank() + 1
            nproc = mpi_comm.Get_size()

        self.nproc, self.rank = nproc, rank


        # import the fortran subroutines
        from f90.interact_between_elems import \
                copy_grid2buf_f90, \
                interact_between_elems_buf_f90, \
                interact_between_elems_inner_f90

        self.copy_grid2buf_f90 = copy_grid2buf_f90
        self.interact_buf_f90 = interact_between_elems_buf_f90
        self.interact_inner_f90 = interact_between_elems_inner_f90


        # load the index tables from a netCDF file
        dpath = os.path.expanduser('~/.pykgm/parallel/ne%d_ngq%d_nproc%d' % (ne, ngq, nproc))
        fpath = dpath + '/rank%d.nc' % (rank)
        nc_grp = nc.Dataset(fpath, 'r', format='NETCDF4')

        assert nc_grp.N == ne
        assert nc_grp.ngq == ngq
        assert nc_grp.rank == rank
        assert nc_grp.nproc == nproc
        self.nelem = nc_grp.nelem
        self.nelem_outer = nc_grp.nelem_outer
        self.buf_size = nc_grp.buf_size
        size_sche = len(nc_grp.dimensions['size_sche'])
        size_grid2buf = len(nc_grp.dimensions['size_grid2buf'])
        size_outer = len(nc_grp.dimensions['size_outer'])
        size_inner = len(nc_grp.dimensions['size_inner'])


        self.send_sche = numpy.zeros((4,size_sche), 'i4', order='F')
        self.recv_sche = numpy.zeros((4,size_sche), 'i4', order='F')
        self.grid2buf = numpy.zeros((3,size_grid2buf), 'i4', order='F')
        self.buf2grid = numpy.zeros((3,size_outer), 'i4', order='F')
        self.mvp_buf = numpy.zeros((4,size_outer), 'i4', order='F')
        self.mvp_inner = numpy.zeros((3,4,size_inner), 'i4', order='F')
        self.elem_coord = numpy.zeros((3,self.nelem), 'i4', order='F')

        self.send_sche[:] = nc_grp.variables['send_sche'][:]
        self.recv_sche[:] = nc_grp.variables['recv_sche'][:]
        self.grid2buf[:] = nc_grp.variables['grid2buf'][:]
        self.buf2grid[:] = nc_grp.variables['buf2grid'][:]
        self.mvp_buf[:] = nc_grp.variables['mvp_buf'][:]
        self.mvp_inner[:] = nc_grp.variables['mvp_inner'][:]
        self.elem_coord[:] = nc_grp.variables['elem_coord'][:]
        self.nelems = nc_grp.variables['nelems'][:]




    def set_variables(self, var_list):
        mpi_comm = self.mpi_comm
        buf_size = self.buf_size
        send_sche = self.send_sche
        recv_sche = self.recv_sche


        for var in var_list:
            ngq, ngq, nlev, nelem = var.shape
            assert ngq == self.ngq
            assert nelem == self.nelem

        num_var = len(var_list)
        dtype = var_list[0].dtype


        # allocate the MPI buffer
        buf = numpy.zeros((nlev, buf_size, num_var), dtype, order='F')


        # initialize nonblock MPI requests
        req_list = {'send':list(), 'recv':list()}
        sl_list = {'send':list(), 'recv':list()}
        subbuf_list = {'send':list(), 'recv':list()}

        for i in xrange( send_sche.shape[-1] ):
            proc, start, end, size = send_sche[:,i]
            sl = slice(start-1, end)
            subbuf = numpy.zeros(buf[:,sl,:].shape, dtype, order='F')
            req_list['send'].append( mpi_comm.Send_init(subbuf, proc-1) )
            sl_list['send'].append(sl)
            subbuf_list['send'].append(subbuf)

        for i in xrange( recv_sche.shape[-1] ):
            proc, start, end, size = recv_sche[:,i]
            sl = slice(start-1, end)
            subbuf = numpy.zeros(buf[:,sl,:].shape, dtype, order='F')
            req_list['recv'].append( mpi_comm.Recv_init(subbuf, proc-1) )
            sl_list['recv'].append(sl)
            subbuf_list['recv'].append(subbuf)


        # assign as the global variables
        self.nlev = nlev
        self.var_list = var_list
        self.buf = buf
        self.req_list = req_list
        self.sl_list = sl_list
        self.subbuf_list = subbuf_list




    def start_buf_exchange(self):
        grid2buf = self.grid2buf
        buf = self.buf
        var_list = self.var_list
        req_list = self.req_list
        sl_list = self.sl_list
        subbuf_list = self.subbuf_list
        copy_grid2buf_f90 = self.copy_grid2buf_f90


        # copy: grid -> buf
        for var_idx, var in enumerate(var_list):
            copy_grid2buf_f90(var_idx+1, grid2buf, buf, var)
    
        # start the nonblock sends
        for req, sl, subbuf in \
                zip(req_list['send'], sl_list['send'], subbuf_list['send']): 
            subbuf[:,:,:] = buf[:,sl,:]
            req.Start()

        # start the nonblock recvs
        for req in req_list['recv']: req.Start()
    
    

    
    def wait_buf_exchange(self):
        req_list = self.req_list
        sl_list = self.sl_list
        subbuf_list = self.subbuf_list
        buf = self.buf


        for req in req_list['send']: req.Wait()
        for req in req_list['recv']: req.Wait()

        for sl, subbuf in zip(sl_list['recv'], subbuf_list['recv']): 
            buf[:,sl,:] = subbuf[:,:,:]




    def interact_between_elems_buf(self):
        buf2grid = self.buf2grid
        mvp_buf = self.mvp_buf
        buf = self.buf
        var_list = self.var_list
        interact_buf_f90 = self.interact_buf_f90


        for var_idx, var in enumerate(var_list):
            interact_buf_f90(var_idx+1, buf2grid, mvp_buf, buf, var)




    def interact_between_elems_inner(self):
        mvp_inner = self.mvp_inner
        var_list = self.var_list
        interact_inner_f90 = self.interact_inner_f90


        for var_idx, var in enumerate(var_list):
            interact_inner_f90(mvp_inner, var)





if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() + 1
    nproc = comm.Get_size()     # 4


    ne, ngq = 6, 4

    if rank == 1:
        interact = InteractBetweenElems(ne, ngq, comm)
