from __future__ import division
import numpy
from numpy import pi, sin, cos, tan, arctan
from tvtk.api import tvtk
from mayavi.sources.vtk_data_source import VTKDataSource
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi import mlab

from pykgm.trial.spectral_element.quadrature import gausslobatto
#from quadrature import gausslobatto



def get_data_source_cubed_sphere_2d(N, ngll):
    '''
    face numbers on the cube
     ---
    | 6 |
     ---
     ---   ---   ---   ---
    | 1 | | 2 | | 3 | | 4 |
     ---   ---   ---   ---
     ---
    | 5 |
     ---
    '''
    gll_pts, gll_wts = gausslobatto(ngll-1)
    da = (gll_pts + 1)*pi/(4*N)

    d_angles = numpy.zeros((ngll-1)*N + 1, 'f8')     # delta_angle list
    d_angles[:ngll] = da
    for i in xrange(1,N):
        d_angles[(ngll-1)*i+1:(ngll-1)*(i+1)+1] = d_angles[(ngll-1)*i] + da[1:]

    '''
    print gll_pts
    print da
    print d_angles
    print pi/2
    '''


    
    #--------------------------------------------------------------------------
    # points for the VTK unstructured grid
    #--------------------------------------------------------------------------
    size_points = 4*((ngll-1)*N)*((ngll-1)*N+1) + 2*((ngll-1)*N-1)**2
    points = numpy.zeros((size_points, 3), 'f8')
    point_idxs = numpy.zeros(((ngll-1)*N+1, (ngll-1)*N+1, 6), 'i4', order='F')


    #-----------------------------------------------
    # face 1, 2, 3, 4
    for face in [1,2,3,4]:
        for gj, beta in enumerate( -pi/4 + d_angles ):
            for gi, alpha in enumerate( -pi/4 + d_angles[:-1] ):
                lon = alpha + (face-1)/2 * pi
                lat = arctan( tan(beta)*cos(alpha) )

                x = cos(lat)*cos(lon)
                y = cos(lat)*sin(lon)
                z = sin(lat)

                idx_face = (ngll-1)*N*((ngll-1)*N+1) * (face-1)
                idx_gj = (ngll-1)*N * gj
                idx = idx_face + idx_gj + gi
                points[idx,:] = x, y, z
                point_idxs[gi,gj,face-1] = idx


    #-----------------------------------------------
    # face 5, 6
    for face in [5,6]:
        for gj, beta in enumerate( -pi/4 + d_angles[1:-1] ):
            for gi, alpha in enumerate( -pi/4 + d_angles[1:-1] ):
                if face == 5:
                    if beta > 0 : 
                        lon = arctan( tan(alpha) / tan(beta) )
                        lat = - arctan( cos(lon) / tan(beta) )
                    elif beta < 0 : 
                        lon = arctan( tan(alpha) / tan(beta) ) + pi
                        lat = - arctan( cos(lon) / tan(beta) )
                    elif beta == 0:
                        lon = pi/2
                        lat = - (pi/2 - alpha)

                elif face == 6:
                    if beta > 0:
                        lon = - arctan( tan(alpha) / tan(beta) ) - pi
                        lat = - arctan( cos(lon) / tan(beta) )
                    elif beta < 0:
                        lon = - arctan( tan(alpha) / tan(beta) )
                        lat = - arctan( cos(lon) / tan(beta) )
                    elif beta == 0:
                        lon = pi/2
                        lat = pi/2 - alpha


                x = cos(lat)*cos(lon)
                y = cos(lat)*sin(lon)
                z = sin(lat)

                idx_face = 4*(ngll-1)*N*((ngll-1)*N+1) + \
			               ( ((ngll-1)*N-1)**2 ) * (face-5)
                idx_gj = ( (ngll-1)*N-1 ) * gj
                idx = idx_face + idx_gj + gi
                points[idx,:] = x, y, z
                point_idxs[gi+1,gj+1,face-1] = idx


    '''
    for face in [1,2,3,4,5,6]:
        print 'face ', face
        for j in xrange( (ngll-1)*N, -1, -1 ):
            print point_idxs[:,j,face-1]
    '''



    #--------------------------------------------------------------------------
    # cells for the VTK unstructured grid
    #--------------------------------------------------------------------------
    size_cells = 6*((ngll-1)*N)**2
    cells = numpy.zeros((size_cells, 4), 'i4')

    # extended the point_idxs 
    point_idxs[-1,:,0] = point_idxs[0,:,1]
    point_idxs[-1,:,1] = point_idxs[0,:,2]
    point_idxs[-1,:,2] = point_idxs[0,:,3]
    point_idxs[-1,:,3] = point_idxs[0,:,0]
    point_idxs[1:,0,4] = point_idxs[::-1,0,2][1:]
    point_idxs[-1,1:,4] = point_idxs[::-1,0,1][1:]
    point_idxs[:-1,-1,4] = point_idxs[:-1,0,0]
    point_idxs[0,:-1,4] = point_idxs[:-1,0,3]
    point_idxs[:-1,0,5] = point_idxs[:-1,-1,0]
    point_idxs[-1,:-1,5] = point_idxs[:-1,-1,1]
    point_idxs[1:,-1,5] = point_idxs[::-1,-1,2][1:]
    point_idxs[0,1:,5] = point_idxs[::-1,-1,3][1:]

    '''
    print '-'*47
    for face in [1,2,3,4,5,6]:
        print 'face ', face
        for j in xrange( (ngll-1)*N, -1, -1 ):
            print point_idxs[:,j,face-1]
    '''


    # Cell: VTK QUAD(9) type 
    for face in [1,2,3,4,5,6]:
        for gj in xrange( (ngll-1)*N ):
            for gi in xrange( (ngll-1)*N ):
                idx_face = (ngll-1)*N*(ngll-1)*N * (face-1)
                idx_gj = (ngll-1)*N * gj
                idx = idx_face + idx_gj + gi

                cells[idx,:] = point_idxs[gi,gj,face-1], \
                               point_idxs[gi+1,gj,face-1], \
                               point_idxs[gi+1,gj+1,face-1], \
                               point_idxs[gi,gj+1,face-1]


    '''
    print '-'*47
    for i in xrange( size_cells ):
        print i, cells[i,:]
    '''



    #--------------------------------------------------------------------------
    # VTK unstructured grid for the cubed-sphere grid
    #--------------------------------------------------------------------------
    # VTK data source on the unstructured grid
    ugrid = tvtk.UnstructuredGrid(points=points)
    cell_types = tvtk.Quad().cell_type      # VTK_QUAD == 9
    ugrid.set_cells(cell_types, cells)
    data_source = VTKDataSource(data=ugrid)

    return data_source


@mlab.show
@mlab.animate(delay=20)
def mlab_animate():
    while True:
        for da in numpy.arange(0, 360, 0.1):
            mlab.view(127-da, 90-38)	# korea centered
            yield



if __name__ == '__main__':
    N, ngll = 4, 4
    csgrid_src = get_data_source_cubed_sphere_2d(N, ngll)

    # cubed-sphere grid
    fig = mlab.figure(bgcolor=(0.48, 0.48, 0.48), fgcolor=(0, 0, 0), size=(800,800))
    csgrid_wire = mlab.pipeline.surface(csgrid_src, representation='wireframe', color=(0,0,0))
    csgrid_surf = mlab.pipeline.surface(csgrid_src, representation='surface', color=(1,1,1), opacity=0.7)
    csgrid_surf.actor.property.backface_culling = True

    # coastline
    coastline_src = BuiltinSurface(source='earth')
    coastline_src.data_source.on_ratio = 2
    coastline = mlab.pipeline.surface(coastline_src, color=(0,0,0))

    #mlab.view(0,90)
    #mlab.show()
    mlab_animate()
    
