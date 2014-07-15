from __future__ import division

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from numpy.random import rand



#points = np.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
points = np.array([[0.1,0.2], [1.5,0.3], [0.4,1.1], [0.9,1.0]])

vor = Voronoi(points)
print 'points\n', vor.points
print 'vertices\n', vor.vertices
print 'ridge_points\n', vor.ridge_points
print 'ridge_vertices\n', vor.ridge_vertices
print 'regions\n', vor.regions
print 'point_region\n', vor.point_region


# plot
#voronoi_plot_2d(vor)

for pt in points:
    plt.plot(pt[0], pt[1], '.')

print '-'*47
for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex>=0):
        print ''
        print simplex, vor.vertices[simplex]
        print vor.vertices[simplex,0], vor.vertices[simplex,1]
        plt.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')

print '-'*47
center = points.mean(axis=0)
print 'center', center
plt.plot(center[0], center[1], '*')

for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.any(simplex < 0):
        print ''
        print 'simplex', simplex, simplex[simplex >= 0], simplex[simplex >= 0][0]
        print 'pointidx', pointidx

        i = simplex[simplex >= 0][0] # finite end Voronoi vertex
        t = points[pointidx[1]] - points[pointidx[0]] # tangent

        print 'tangent', t

        t /= np.linalg.norm(t)

        print 'tangent/norm', t

        n = np.array([-t[1], t[0]]) # normal
        midpoint = points[pointidx].mean(axis=0)
        plt.plot(midpoint[0], midpoint[1], 'x')


        far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
        plt.plot([vor.vertices[i,0], far_point[0]],
                 [vor.vertices[i,1], far_point[1]], 'k--')

plt.show(True)
