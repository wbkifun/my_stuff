from __future__ import division
import numpy
import os
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size() 



from cube_partition import CubePartition
from interact_between_elems import InteractBetweenElems


# setup
ne, ngq = 6, 4
# nproc, rank = 10, 1

cube = CubePartition(ne, ngq, nproc, rank+1)
cube.save_netcdf()

interact = InteractBetweenElems(ne, ngq, comm)
nelems = interact.nelems
nelem = nelems[rank]
elem_coord = interact.elem_coord


f = numpy.zeros((ngq,ngq,1,nelem), 'f8', order='F')
f[:] = numpy.random.rand((ngq*ngq*nelem)).reshape(ngq,ngq,1,nelem)


interact.set_variables([f])
interact.start_buf_exchange()
interact.interact_between_elems_inner()
interact.wait_buf_exchange()
interact.interact_between_elems_buf()


if rank != 0:
    comm.send([f, elem_coord], dest=0, tag=1)


else:
    print 'nelems', nelems

    #-----------------------------------------------------------
    # recieve f array and elem_coord
    #-----------------------------------------------------------
    f_list = [f]
    elem_coord_list = [elem_coord]

    for src in xrange(1,nproc):
        f2, elem_coord2 = comm.recv(source=src, tag=1)
        f_list.append(f2)
        elem_coord_list.append(elem_coord2)


    #-----------------------------------------------------------
    # entire array
    #-----------------------------------------------------------
    tf = numpy.zeros((ngq,ngq,ne,ne,6), 'f8', order='F')

    for nelem, elem_coord, f in zip(nelems, elem_coord_list, f_list):
        for ielem in xrange(nelem):
            ei, ej, face = elem_coord[:,ielem]
            tf[:,:,ei-1,ej-1,face-1] = f[:,:,0,ielem]


    #-----------------------------------------------------------
    # verify the same values of the MVP (Multi-Value Point)
    #-----------------------------------------------------------
    basepath = os.path.expanduser('~/.pykgm/cs_grid/')
    dpath = basepath + 'ne%d_ngq%d/'%(ne,ngq)
    mvp_coord = numpy.load(dpath+'mvp_coord.npy')
    mvp_num = numpy.load(dpath+'mvp_num.npy')


    for face in xrange(6):
        for ej in xrange(ne):
            for ei in xrange(ne):
                for gj in xrange(ngq):
                    for gi in xrange(ngq):
                        val = tf[gi,gj,ei,ej,face]

                        for mi in xrange(mvp_num[gi,gj,ei,ej,face]-1):
                            mgi,mgj,mei,mej,mface = mvp_coord[:,mi,gi,gj,ei,ej,face]

                            assert_aae(tf[mgi-1,mgj-1,mei-1,mej-1,mface-1], val, 15)
