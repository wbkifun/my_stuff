from __future__ import division
import numpy
import pickle
import os

from space_filling_curve import SpaceFillingCurve



class CubePartition(object):
    def __init__(self, N, ngq, nproc, rank):
        assert ngq >= 2

        self.N = N
        self.ngq = ngq
        self.nproc = nproc
        self.rank = rank


        #======================================================================
        # set the variables
        #======================================================================
        #----------------------------------------------------------------------
        self.cube_proc = numpy.zeros((N,N,6), 'i4', order='F')  # process number
        self.nelems = numpy.zeros(nproc, 'i4')                  # number of elements

        # partitioning the cube using SFC
        self.set_cube_proc_sfc()
        self.nelem = self.nelems[rank-1]


        #----------------------------------------------------------------------
        self.mvp_coord = numpy.zeros((5,3,ngq,ngq,N,N,6), 'i4', order='F') 
        self.mvp_num = numpy.zeros((ngq,ngq,N,N,6), 'i4', order='F') 
        self.mvp_proc = numpy.ones((3,ngq,ngq,N,N,6), 'i4', order='F')*(-1)

        # load the mvp_coord, mvp_num
        # set the mvp_proc
        self.set_mvp_proc()


        #----------------------------------------------------------------------
        # coord. of local elements
        self.elem_coord = numpy.zeros((3,self.nelem), 'i4', order='F')

        # find the number of outer elements
        # set the elem_coord of inner elements
        # select the start Gauss-Quadrature point
        self.nelem_outer, self.start_gq_coord = self.select_start_gq_coords()


        #----------------------------------------------------------------------
        # find the outmost Gauss-quadrature points along anti-clockwise direction
        # list type: [(gi,gj,ei,ej,face), ...]
        self.outmost_gq_coord = self.find_outmost_gq_coord_anti_clockwise()


        #----------------------------------------------------------------------
        # generate the information for MPI buffer
        # buf_sizes: {'send':, 'recv':, 'total':}
        # send_buf_coord: [[(gi,gj,ei,ej,face), [proc,proc]]...]
        # coord2bufidx: {(gi,gj,ei,ej,face):#, ...]
        self.buf_sizes, self.send_buf_coord, self.coord2bufidx = self.gen_buf_info()

        # set the elem_coord of outer_element
        self.set_elem_coord_outer()

        # generate the table to find the mapping as (ei,ej,face) -> ielem
        self.coord2ielem = self.gen_coord2ielem()


        #======================================================================
        # output arrays to interact between elements on MPI environment
        #======================================================================
        # generate the MPI schedule table
        # target rank, start index, end index, size
        # shape (4,lengh)
        self.send_schedule, self.recv_schedule = self.gen_mpi_schedule()

        # grid -> buffer
        # set the coordinate table to copy grid->buf: [(gi,gj,ielem), ...]
        send_buf_size = self.buf_sizes['send']
        self.grid2buf_coord = numpy.zeros((3,send_buf_size), 'i4', order='F')
        self.set_grid2buf_coord()

        # buffer -> grid
        # generate the MVP(Multi Value Points) indices on the MPI buffer
        outmost_size = len(self.outmost_gq_coord)
        self.buf2grid_coord = numpy.zeros((3,outmost_size), 'i4', order='F')
        self.mvp_idxs_buf = numpy.ones((4,outmost_size), 'i4', order='F')*(-1)
        self.set_mvp_idxs_buf()

        # in the grid
        # generate the MVP(Multi Value Points) indices on the inner grid
        # shape (3,4,length)
        self.mvp_idxs_inner = self.gen_mvp_idxs_inner()




    def set_cube_proc_sfc(self):
        '''
        partitioning the cube using the space-filling curves
        chane the face numbers
             -                      -
            |3|                    |6|
         -   -   -   -          -   -   -   -
        |4| |1| |2| |6|   =>   |4| |1| |2| |3|
         -   -   -   -          -   -   -   -
            |5|                    |5|
             -                      -
        '''

        N = self.N
        ngq = self.ngq
        nproc = self.nproc

        sfc = SpaceFillingCurve(N, ngq, nproc)
        self.cube_proc[:,:,0] = sfc.cube_proc[:,:,0]
        self.cube_proc[:,:,1] = sfc.cube_proc[:,:,1]
        self.cube_proc[:,:,2] = sfc.cube_proc[:,:,5]
        self.cube_proc[:,:,3] = sfc.cube_proc[:,:,3]
        self.cube_proc[:,:,4] = sfc.cube_proc[:,:,4]
        self.cube_proc[:,:,5] = sfc.cube_proc[:,:,2]
        self.nelems[:] = sfc.nelems




    def set_mvp_proc(self):
        '''
        1. load the pre-processed files
        mvp_coord   (6,3,ngq,ngq,N,N,6): coordinates of the Multi-Valued Points
        mvp_num     (ngq,ngq,N,N,6): number of the multi-valued points

        2. re-generate
        mvp_proc    (3,ngq,ngq,N,N,6): process number of the multi-valued points
        '''

        N = self.N
        ngq = self.ngq
        mvp_coord = self.mvp_coord
        mvp_num = self.mvp_num
        cube_proc = self.cube_proc
        mvp_proc = self.mvp_proc


        #---------------------------------------------------------
        # load the mvp_coord, mvp_num
        #---------------------------------------------------------
        basepath = os.path.expanduser('~/.pykgm/cs_grid/')
        dpath = basepath + 'ne%d_ngq%d/'%(N,ngq)

        '''
        if rank == 1:
            if not os.path.exists(dpath+'mvp_coord.npy'):
                from pykgm.share.cs_grid_structure import save_mvp_coord
                save_mvp_coord(N, ngq)
        comm.Barrier()
        '''

        mvp_coord[:] = numpy.load(dpath+'mvp_coord.npy')
        mvp_num[:] = numpy.load(dpath+'mvp_num.npy')


        #---------------------------------------------------------
        # generate the mvp_proc
        #---------------------------------------------------------
        for face in xrange(6):
            for ej in xrange(N):
                for ei in xrange(N):
                    for gj in xrange(ngq):
                        for gi in xrange(ngq):
                            for mi in xrange(mvp_num[gi,gj,ei,ej,face]-1):
                                mgi,mgj,mei,mej,mface = mvp_coord[:,mi,gi,gj,ei,ej,face]
                                mvp_proc[mi,gi,gj,ei,ej,face] = cube_proc[mei-1,mej-1,mface-1]




    def select_start_gq_coords(self):
        '''
        find the number of outer elements
        set the elem_coord of the inner elements
        select the start Gauss-quadrature point
        '''

        N = self.N
        ngq = self.ngq
        rank = self.rank
        cube_proc = self.cube_proc
        mvp_proc = self.mvp_proc
        elem_coord = self.elem_coord


        ielem_outer = 0
        ielem_inner = 0
        max_num_neighbor_proc = 0

        for face in xrange(6):
            for ej in xrange(N):
                for ei in xrange(N):
                    if cube_proc[ei,ej,face] == rank:
                        is_outer_elem = False

                        for gj in xrange(ngq):
                            for gi in xrange(ngq):
                                proc_set = set( [rank] + \
                                        list(mvp_proc[:,gi,gj,ei,ej,face]) )
                                if -1 in proc_set: proc_set.remove(-1)

                                if len(proc_set) >= 2:
                                    is_outer_elem = True

                                if len(proc_set) > max_num_neighbor_proc:
                                    max_num_neighbor_proc = len(proc_set)
                                    start_gq_coord = (gi,gj,ei,ej,face)

                        if is_outer_elem:
                            ielem_outer += 1
                        else:
                            elem_coord[:,ielem_inner] = (ei+1,ej+1,face+1)
                            ielem_inner += 1

        # rearange elem_coord
        assert ielem_outer + ielem_inner == self.nelem
        elem_coord[:,ielem_outer:] = elem_coord[:,:ielem_inner]
        elem_coord[:,:ielem_outer] = -1


        return ielem_outer, start_gq_coord
                            



    def find_outmost_gq_coord_anti_clockwise(self):
        '''
        find the outmost Gauss-quadrature points along anti-clockwise direction
        '''

        ngq = self.ngq
        rank = self.rank
        start_gq = self.start_gq_coord
        mvp_coord = self.mvp_coord
        mvp_num = self.mvp_num
        mvp_proc = self.mvp_proc


        outmost_gq_coord = []
        now_gq = start_gq

        while True:
            # append the outmost_gq
            # jump the last edge
            gi, gj, ei, ej, face = now_gq
            outmost_gq_coord.append( (gi+1,gj+1,ei+1,ej+1,face+1) )
            for mi, proc in zip([2,1,0], mvp_proc[::-1,gi,gj,ei,ej,face]):
                if proc == rank:
                    mgi,mgj,mei,mej,mface = mvp_coord[:,mi,gi,gj,ei,ej,face]
                    outmost_gq_coord.append( (mgi,mgj,mei,mej,mface) )
                    now_gq = (mgi-1,mgj-1,mei-1,mej-1,mface-1)

            
            # move to the next gq point
            gi, gj, ei, ej, face = now_gq
            if mvp_num[now_gq] in [3,4]:       # edge point
                elem_side_type = {(0,0):'a', (ngq-1,0):'b', \
                                  (ngq-1,ngq-1):'c', (0,ngq-1):'d'}[(gi,gj)]
            
            if elem_side_type == 'a': next_gq = (gi+1,gj,ei,ej,face)
            elif elem_side_type == 'b': next_gq = (gi,gj+1,ei,ej,face)
            elif elem_side_type == 'c': next_gq = (gi-1,gj,ei,ej,face)
            elif elem_side_type == 'd': next_gq = (gi,gj-1,ei,ej,face)

            now_gq = next_gq


            # exit condition
            if next_gq == start_gq: break


        return outmost_gq_coord




    def gen_buf_info(self):
        '''
        generate the information for MPI buffer
        buf_sizes: {'send':, 'recv':, 'total':}
        send_buf_coord: [[(gi,gj,ei,ej,face), [proc,proc]]...]
        coord2bufidx: {(gi,gj,ei,ej,face):#, ...]
        '''

        rank = self.rank
        mvp_coord = self.mvp_coord
        mvp_num = self.mvp_num
        mvp_proc = self.mvp_proc
        outmost_gq_coord = self.outmost_gq_coord


        #----------------------------------------------------------------------
        # generate the send buffer coordinate
        #----------------------------------------------------------------------
        coord2bufidx = dict()
        send_coord_head, send_coord_tail = list(), list()

        # end-point send_proc set
        gi_b0, gj_b0, ei_b0, ej_b0, face_b0 = outmost_gq_coord[-1]
        send_procs_b0 = []
        for proc in mvp_proc[:,gi_b0-1,gj_b0-1,ei_b0-1,ej_b0-1,face_b0-1]:
            if proc not in [-1, rank]: 
                send_procs_b0.append(proc)
        set_proc_b0 = set(send_procs_b0)


        # find the sequence and offset to make the successive buffer region
        seq_tail = len(outmost_gq_coord) + 1
        send_offset = 0
        send_duplicated = 0
        check_intersect = True

        for seq_send, (gi,gj,ei,ej,face) in enumerate(outmost_gq_coord):
            seq = seq_send + 1
            coord = (gi,gj,ei,ej,face)

            send_procs = []
            for proc in mvp_proc[:,gi-1,gj-1,ei-1,ej-1,face-1]:
                if proc not in [-1, rank]:
                    send_procs.append(proc)
            set_proc = set(send_procs)
            intersect = set_proc.intersection(set_proc_b0)
            set_diff = set_proc - intersect

            if check_intersect and len(intersect) > 0:
                if len(set_diff) == 0:
                    coord2bufidx[coord] = seq_tail
                    send_coord_tail.append( [coord, list(intersect)] )
                    send_offset += 1
                else:
                    coord2bufidx[coord] = seq
                    send_coord_head.append( [coord, list(set_diff)] )
                    send_coord_tail.append( [coord, list(intersect)] )
                    send_duplicated += 1

                seq_tail += 1

            else:
                coord2bufidx[coord] = seq
                send_coord_head.append( [coord, list(set_proc)] )
                check_intersect = False


        # apply the send_offset
        if send_offset != 0:
            for coord in coord2bufidx.keys():
                coord2bufidx[coord] -= send_offset


        send_buf_coord = send_coord_head + send_coord_tail
        send_buf_size = len(send_buf_coord)
        assert send_buf_size == len(outmost_gq_coord) + send_duplicated
        if self.nproc != 1:
            assert len(coord2bufidx) == len(outmost_gq_coord)


        #----------------------------------------------------------------------
        # generate the recv buffer coordinate
        #----------------------------------------------------------------------
        recv_coord_seq = dict()
        seq_recv = 0

        for (gi,gj,ei,ej,face), send_procs in send_buf_coord:
            for mi in xrange(mvp_num[gi-1,gj-1,ei-1,ej-1,face-1] - 1):
                proc = mvp_proc[mi,gi-1,gj-1,ei-1,ej-1,face-1]

                if proc in send_procs:
                    mcoord = tuple( mvp_coord[:,mi,gi-1,gj-1,ei-1,ej-1,face-1] )

                    if mcoord not in recv_coord_seq.keys():
                        recv_coord_seq[mcoord] = seq_recv
                        seq_recv += 1
    
        recv_buf_size = len(recv_coord_seq)
        buf_size = send_buf_size + recv_buf_size


        # append the recv's reverse sequence to coord2bufidx dictionary
        for coord, recv_seq in recv_coord_seq.items():
            coord2bufidx[coord] = buf_size - recv_seq

        if self.nproc != 1:
            assert len(coord2bufidx) == buf_size - send_duplicated


        buf_sizes = {'send':send_buf_size, 'recv':recv_buf_size, 'total':buf_size}

        return buf_sizes, send_buf_coord, coord2bufidx




    def set_elem_coord_outer(self):
        '''
        set the elem_coord of the outer elements
        '''

        nelem_outer = self.nelem_outer
        send_buf_coord = self.send_buf_coord
        elem_coord = self.elem_coord


        elem_coord_tmp = list()
        ielem_outer = 0

        for (gi,gj,ei,ej,face), send_procs in send_buf_coord:
            if ielem_outer == nelem_outer: break

            coord = (ei,ej,face)
            
            if coord not in elem_coord_tmp:
                elem_coord_tmp.append(coord)
                elem_coord[:,ielem_outer] = coord
                ielem_outer += 1




    def gen_coord2ielem(self):
        '''
        table to find the mapping as (ei,ej,face) -> ielem
        '''
        nelem = self.nelem
        elem_coord = self.elem_coord


        coord2ielem = dict()
        for ielem in xrange(nelem):
            (ei,ej,face) = elem_coord[:,ielem]
            coord2ielem[(ei,ej,face)] = ielem + 1

        return coord2ielem




    def gen_mpi_schedule(self):
        '''
        generate the MPI schedule table
        '''

        rank = self.rank
        cube_proc = self.cube_proc
        mvp_num = self.mvp_num
        mvp_proc = self.mvp_proc
        buf_sizes = self.buf_sizes
        send_buf_coord = self.send_buf_coord
        coord2bufidx = self.coord2bufidx

        
        #----------------------------------------------------------------------
        # generate the send schedule
        #----------------------------------------------------------------------
        send_sche = dict()

        for i, [(gi,gj,ei,ej,face), send_procs] in enumerate(send_buf_coord):
            idx = i + 1
            
            for proc in send_procs:
                if proc not in send_sche.keys():
                    # [start_idx, end_idx]
                    send_sche[proc] = [idx, idx]

                else:
                    if idx < send_sche[proc][0]:
                        send_sche[proc][0] = idx

                    elif idx > send_sche[proc][1]:
                        send_sche[proc][1] = idx


        # send schedule summerize
        send_procs = sorted( send_sche.keys() )
        send_schedule = numpy.zeros((4, len(send_procs)), 'i4', order='F')
        for i, proc in enumerate(send_procs):
            start_idx, end_idx = send_sche[proc]
            size = end_idx - start_idx + 1
            send_schedule[:,i] = (proc, start_idx, end_idx, size)
                
        if self.nproc != 1:
            assert send_schedule[1,:].min() == 1
            assert send_schedule[2,:].max() == buf_sizes['send']


        #----------------------------------------------------------------------
        # generate the recv schedule
        #----------------------------------------------------------------------
        recv_sche = dict()

        for (gi,gj,ei,ej,face), idx in coord2bufidx.items():
            proc = cube_proc[ei-1,ej-1,face-1]

            if proc != rank:
                if proc not in recv_sche.keys():
                    # [start_idx, end_idx]
                    recv_sche[proc] = [idx, idx]

                else:
                    if idx < recv_sche[proc][0]:
                        recv_sche[proc][0] = idx

                    elif idx > recv_sche[proc][1]:
                        recv_sche[proc][1] = idx


        # recv schedule summerize
        recv_procs = sorted( recv_sche.keys() )
        recv_schedule = numpy.zeros((4, len(recv_procs)), 'i4', order='F')
        for i, proc in enumerate(recv_procs):
            start_idx, end_idx = recv_sche[proc]
            size = end_idx - start_idx + 1
            recv_schedule[:,i] = (proc, start_idx, end_idx, size)
                
        assert recv_schedule[-1,:].sum() == buf_sizes['recv']


        return send_schedule, recv_schedule




    def set_grid2buf_coord(self):
        '''
        generate the coord. table to copy grid -> buffer
        [(gi,gj,ielem), ...]
        '''
        
        send_buf_coord = self.send_buf_coord
        coord2ielem = self.coord2ielem 
        grid2buf_coord = self.grid2buf_coord


        for i, [(gi,gj,ei,ej,face), send_procs] in enumerate(send_buf_coord):
            ielem = coord2ielem[(ei,ej,face)]
            grid2buf_coord[:,i] = (gi,gj,ielem)




    def set_mvp_idxs_buf(self):
        '''
        generate the MVP(Multi Value Points) indices on the MPI buffer
        '''

        mvp_coord = self.mvp_coord
        mvp_num = self.mvp_num
        outmost_gq_coord = self.outmost_gq_coord
        coord2ielem = self.coord2ielem
        coord2bufidx = self.coord2bufidx
        buf2grid_coord = self.buf2grid_coord
        mvp_idxs_buf = self.mvp_idxs_buf


        for i, (gi,gj,ei,ej,face) in enumerate(outmost_gq_coord):
            #-----------------------------------------------------
            ielem = coord2ielem[(ei,ej,face)]
            buf2grid_coord[:,i] = (gi,gj,ielem)


            #-----------------------------------------------------
            Nmvp = mvp_num[gi-1,gj-1,ei-1,ej-1,face-1]
            bufidx_list = [ coord2bufidx[(gi,gj,ei,ej,face)] ]

            for mi in xrange(Nmvp-1):
                coord = tuple( mvp_coord[:,mi,gi-1,gj-1,ei-1,ej-1,face-1] )
                bufidx_list.append( coord2bufidx[coord] )

            mvp_idxs_buf[:Nmvp,i] = bufidx_list




    def gen_mvp_idxs_inner(self):
        '''
        generate the MVP(Multi Value Points) indices on the inner grid
        '''

        ngq = self.ngq
        nelem = self.nelem
        mvp_coord = self.mvp_coord
        mvp_num = self.mvp_num
        elem_coord = self.elem_coord
        coord2ielem = self.coord2ielem
        outmost_gq_coord = self.outmost_gq_coord


        mvp_idxs_list = []

        for ielem in xrange(nelem):
            ei, ej, face = elem_coord[:,ielem]

            for gj in xrange(1,ngq+1):
                for gi in xrange(1,ngq+1):
                    if (gi,gj,ei,ej,face) not in outmost_gq_coord:
                        Nmvp = mvp_num[gi-1,gj-1,ei-1,ej-1,face-1]

                        if Nmvp != 1:
                            ielem = coord2ielem[(ei,ej,face)]
                            coord_list = [(gi,gj,ielem)]

                            for mi in xrange(3):
                                mgi,mgj,mei,mej,mface = \
                                        mvp_coord[:,mi,gi-1,gj-1,ei-1,ej-1,face-1]

                                if mgi != -1:
                                    mielem = coord2ielem[(mei,mej,mface)]
                                    coord_list.append( (mgi,mgj,mielem) )
                                else:
                                    coord_list.append( (-1,-1,-1) )

                            mvp_idxs_list.append( coord_list )


        mvp_idxs_inner = numpy.ones((3,4,len(mvp_idxs_list)), 'i4', order='F')*(-1)
        for i, coord_list in enumerate(mvp_idxs_list):
            for j, coord in enumerate(coord_list):
                mvp_idxs_inner[:,j,i] = coord


        return mvp_idxs_inner




    def save_netcdf(self):                                                    
        import netCDF4 as nc

        N, ngq = self.N, self.ngq
        nproc, rank = self.nproc, self.rank

        dpath = os.path.expanduser('~/.pykgm/parallel/ne%d_ngq%d_nproc%d' % (N, ngq, nproc))
        #if not os.path.exists(dpath): os.makedirs(dpath)
        fpath = dpath + '/rank%d.nc' % (rank)
        grp = nc.Dataset(fpath, 'w', format='NETCDF4')


        # attributes
        grp.description = 'parallelization using SFC for the SEM-based dynamical cores on the cubed-sphere grid'
        grp.N = self.N
        grp.ngq = self.ngq
        grp.rank = self.rank
        grp.nproc = self.nproc
        grp.nelem = self.nelem
        grp.nelem_outer = self.nelem_outer
        grp.buf_size = self.buf_sizes['total']

        # dimensions
        grp.createDimension('size_sche', self.send_schedule.shape[-1])
        grp.createDimension('size_grid2buf', self.grid2buf_coord.shape[-1])
        grp.createDimension('size_outer', self.buf2grid_coord.shape[-1])
        grp.createDimension('size_inner', self.mvp_idxs_inner.shape[-1])
        grp.createDimension('nelem', self.nelem)
        grp.createDimension('4', 4)
        grp.createDimension('3', 3)
        grp.createDimension('nproc', self.nproc)

        # variables
        send_sche_nc = grp.createVariable('send_sche', 'i4', ('4','size_sche'))
        recv_sche_nc = grp.createVariable('recv_sche', 'i4', ('4','size_sche'))
        grid2buf_nc = grp.createVariable('grid2buf', 'i4', ('3','size_grid2buf'))
        buf2grid_nc = grp.createVariable('buf2grid', 'i4', ('3','size_outer'))
        mvp_buf_nc = grp.createVariable('mvp_buf', 'i4', ('4','size_outer'))
        mvp_inner_nc = grp.createVariable('mvp_inner', 'i4', ('3','4','size_inner'))
        elem_coord_nc = grp.createVariable('elem_coord', 'i4', ('3', 'nelem'))
        nelems_nc = grp.createVariable('nelems', 'i4', ('nproc',))

        # assign data to variables
        send_sche_nc[:] = self.send_schedule
        recv_sche_nc[:] = self.recv_schedule
        grid2buf_nc[:] = self.grid2buf_coord
        buf2grid_nc[:] = self.buf2grid_coord
        mvp_buf_nc[:] = self.mvp_idxs_buf
        mvp_inner_nc[:] = self.mvp_idxs_inner
        elem_coord_nc[:] = self.elem_coord
        nelems_nc[:] = self.nelems

        grp.close()




if __name__ == '__main__':
    ne, ngq = 6, 4
    nproc, rank = 10, 10

    cube = CubePartition(ne, ngq, nproc, rank)
    print cube.nelem, cube.nelem_outer, numpy.array(cube.start_gq_coord)+1

    print '-'*80
    print 'elem_coord'
    for i in xrange(cube.nelem):
        print i+1, cube.elem_coord[:,i]

    print '-'*80
    print 'MPI schedule'
    print 'buf_sizes', cube.buf_sizes

    print 'send schedule'
    print cube.send_schedule.T

    print 'recv schedule'
    print cube.recv_schedule.T


    print '-'*80
    print "grid2buf_coord"
    for i in xrange( cube.grid2buf_coord.shape[-1] ):
        print i+1, cube.grid2buf_coord[:,i]

    print '-'*80
    print "buf2grid_coord"
    for i in xrange( cube.buf2grid_coord.shape[-1] ):
        print i+1, cube.buf2grid_coord[:,i]

    print '-'*80
    print "mvp_idx_buf"
    for i in xrange( cube.mvp_idxs_buf.shape[-1] ):
        print i+1, cube.mvp_idxs_buf[:,i]

    print '-'*80
    print "mvp_idx_inner"
    for i in xrange( cube.mvp_idxs_inner.shape[-1] ):
        print ''
        for j in xrange(4):
            print i+1, j+1, cube.mvp_idxs_inner[:,j,i]

    cube.save_netcdf()
