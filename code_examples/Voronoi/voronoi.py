from __future__ import division

import numpy as np
from scipy.spatial import voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from numpy.random import rand



points = np.array([[0,0], [0,2], [2,2]])
#points = rand(10).reshape((5,2))
'''
points = np.array(
        [[ 0.22403616,  0.39571284],
         [ 0.85056796,  0.34780449],
          [ 0.36541502,  0.30568376],
           [ 0.38902664,  0.80566618],
            [ 0.66418705,  0.08285052]])
'''

vor = voronoi(points)
print 'points\n', vor.points
print 'vertices\n', vor.vertices
print 'ridge_points\n', vor.ridge_points
print 'ridge_vertices\n', vor.ridge_vertices
print 'regions\n', vor.regions
print 'point_region\n', vor.point_region


voronoi_plot_2d(vor)
plt.show()
