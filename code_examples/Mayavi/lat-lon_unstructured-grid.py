from __future__ import division
import numpy
from numpy import pi, sin, cos, exp



#----------------------------------------------
# lat-lon coordinates
nx = 50
ny = nx
lats = numpy.linspace(-pi/2, pi/2, ny)
lons = numpy.linspace(0, 2*pi, nx)


#----------------------------------------------
# scalar variable
sf = numpy.zeros((nx,ny), 'f4', order='F')

for j, phi in enumerate(lats):
	for i, theta in enumerate(lons):
		sf[i,j] = exp( -(((theta-pi)*cos(phi))**2 + phi**2)/(pi/50) )	# gaussian


#----------------------------------------------
# plot
#----------------------------------------------
from mayavi import mlab
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.sources.vtk_data_source import VTKDataSource
from tvtk.api import tvtk


#mlab.options.offscreen = True

fig = mlab.figure(bgcolor=(0, 0, 0), size=(800,800))


#----------------------------------------------
# unstructured grid for the lat-lon grid
points = numpy.zeros((nx*ny,3), 'f4')
cells = numpy.zeros(((nx-1)*(ny-1),4), 'i4')

for j, phi in enumerate(lats):
	for i, theta in enumerate(lons):
		idx = j*nx + i

		x = cos(phi)*cos(theta)
		y = cos(phi)*sin(theta)
		z = sin(phi)
		points[idx,:] = x, y, z

for j in xrange(ny-1):
	for i in xrange(nx-1):
		idx = j*(nx-1) + i
		idx_pt = j*nx + i

		cells[idx,:] = idx_pt, idx_pt+1, idx_pt+nx+1, idx_pt+nx


ugrid = tvtk.UnstructuredGrid(points=points)
cell_types = tvtk.Quad().cell_type	# VTK_QUAD == 9
ugrid.set_cells(cell_types, cells)
ugrid.point_data.scalars = sf.flatten(order='F')
ugrid.point_data.scalars.name = 'scalar field'

field_src = VTKDataSource(data = ugrid)
grid = mlab.pipeline.surface(field_src, color=(0,0,0), representation='wireframe')
field = mlab.pipeline.surface(field_src, colormap='jet', opacity=0.8)


#----------------------------------------------
# coastline
coastline_src = BuiltinSurface(source='earth')
coastline_src.data_source.on_ratio = 2
coastline = mlab.pipeline.surface(coastline_src, color=(1,1,1))


#----------------------------------------------
# show
mlab.view(127, 90-38)
mlab.show()
#mlab.savefig('lat-lon.png')
