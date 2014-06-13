from __future__ import division
import numpy
import os
import sys
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size() 



from cube_partition import CubePartition
from interact_between_elems import InteractBetweenElemsAvg


#------------------------------------------------------------------------------
# setup
#------------------------------------------------------------------------------
try:
    ne, ngq = int(sys.argv[1]), int(sys.argv[2])
except:
    ne, ngq = 6, 4
    if rank==0: print 'default value: ne=%d, ngq=%d' % (ne, ngq)


#------------------------------------------------------------------------------
# check the cubed-sphere grid information files
#------------------------------------------------------------------------------
dpath = os.path.expanduser('~/.pykgm/cs_grid/ne%d_ngq%d/' % (ne,ngq))
if not os.path.exists(dpath+'mvp_coord.npy'):
    from pykgm.share.cs_grid_structure import save_mvp_coord, save_is_uvp
    save_mvp_coord(ne, ngq)
    save_is_uvp(ne, ngq)


#------------------------------------------------------------------------------
# save the index tables as netcdf format
#------------------------------------------------------------------------------
cube = CubePartition(ne, ngq, nproc, rank+1)
cube.save_netcdf(comm)


#------------------------------------------------------------------------------
# create a interact object
#------------------------------------------------------------------------------
interact = InteractBetweenElemsAvg(ne, ngq, comm)
nelems = interact.nelems
nelem = nelems[rank]
ielem2coord = interact.ielem2coord


#------------------------------------------------------------------------------
# generate a test variable
#------------------------------------------------------------------------------
f = numpy.zeros((ngq,ngq,1,nelem), 'f8', order='F')
f[:] = numpy.random.rand((ngq*ngq*nelem)).reshape(ngq,ngq,1,nelem)


#------------------------------------------------------------------------------
# average on the MVP(multi-value points) between elements
#------------------------------------------------------------------------------
if nproc == 1:
    interact.set_variables([f])
    interact.interact_inner_avg()

else:
    interact.set_variables([f])
    interact.start_buf_exchange()
    interact.interact_inner_avg()
    interact.wait_buf_exchange()
    interact.interact_buf_avg()



#------------------------------------------------------------------------------
# gathering the test variables from all MPI processes
#------------------------------------------------------------------------------
if rank != 0:
    comm.send([f, ielem2coord], dest=0, tag=1)

else:
    print 'nproc=%d, nelems %s' % (nproc, nelems)

    #-----------------------------------------------------------
    # recieve f array and ielem2coord
    #-----------------------------------------------------------
    f_list = [f]
    ielem2coord_list = [ielem2coord]

    for src in xrange(1,nproc):
        f2, ielem2coord2 = comm.recv(source=src, tag=1)
        f_list.append(f2)
        ielem2coord_list.append(ielem2coord2)


    #-----------------------------------------------------------
    # entire array
    #-----------------------------------------------------------
    tf = numpy.zeros((ngq,ngq,ne,ne,6), 'f8', order='F')

    for nelem, ielem2coord, f in zip(nelems, ielem2coord_list, f_list):
        for ielem in xrange(nelem):
            ei, ej, face = ielem2coord[:,ielem]
            tf[:,:,ei-1,ej-1,face-1] = f[:,:,0,ielem]



#------------------------------------------------------------------------------
# verify the same values on the MVP (Multi-Value Point)
#------------------------------------------------------------------------------
if rank == 0:
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
