from __future__ import division

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from numpy.random import rand



def calc_far_vertex(center, vor):
    '''
    calculate the infinite vertex
    refer to http://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html
    '''

    pointidx = vor.ridge_points                 # indices of two points
    simplex  = np.asarray(vor.ridge_vertices)   # indices of two vertices include -1

    if np.any(simplex < 0):
        i = simplex[simplex >= 0][0]            # finite end Voronoi vertex

        t = points[pointidx[1]] - points[pointidx[0]]   # tangent
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]])                     # normal

        midpoint = points[pointidx].mean(axis=0)

        pm = np.sign(np.dot(midpoint - center, n))
        far_point = vor.vertices[i] + pm*n*100

        return far_point

    else:
        return None




def calc_ext_vertex(vertex, midpoint, center, bound_xy):
    x1, y1 = vertex
    x2, y2 = midpoint
    xc, yc = center

    bx1, bx2 = bound_xy[:,0]
    by1, by2 = bound_xy[:,1]

    try:
        slope = (y2-y1)/(x2-x1)

    except ZeroDvisionError:
        x3 = x1
        y3 = by1 if (yc-y1)*(y1-by1) > 0 else by2

        return (x3,y3)


    x3 = bx1 if (xc-x1)*(x1-bx1) > 0 else bx2
    y3 = slope*(x3-x1) + y1

    if not by1 <= y3 <= by2:
        y3 = by1 if (yc-y1)*(y1-by1) > 0 else by2
        x3 = 1/slope*(y3-y1) + x1

    return (x3,y3)




#points = np.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
#points = np.array([[0.1,0.2], [1.5,0.3], [0.4,1.1], [0.9,1.0]])
#points = np.array([[0.73953542, 0.10835294], [0.0428361, 0.96315362], [0.96574911, 0.04137287]])

npt = 4
points = np.random.rand(npt*2).reshape(npt,2)


vor = Voronoi(points)
print 'points\n', vor.points
print 'vertices\n', vor.vertices
print type(vor.vertices), vor.vertices.shape
print vor.vertices[0]
print 'ridge_points\n', vor.ridge_points
print 'Nrd', vor.ridge_points.shape[0]
print 'ridge_vertices\n', vor.ridge_vertices
print 'regions\n', vor.regions
print 'point_region\n', vor.point_region



pts = np.concatenate((points,vor.vertices))
xmin, xmax = pts[:,0].min(), pts[:,0].max()
ymin, ymax = pts[:,1].min(), pts[:,1].max()

bx1, bx2 = xmin-abs(xmax-xmin)*0.1, xmax+abs(xmax-xmin)*0.1
by1, by2 = ymin-abs(ymax-ymin)*0.1, ymax+abs(ymax-ymin)*0.1
bound_xy = np.array([[bx1,by1], [bx2,by2]])


#==========================================================================
# plot
#==========================================================================
#voronoi_plot_2d(vor)

fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(1,1,1)

ax1.plot(points[:,0], points[:,1], 'bo')

#--------------------------------------------------------------------------
# internal ridges
#--------------------------------------------------------------------------
print '-'*47
for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex>=0):
        x1, y1 = vor.vertices[simplex,0], vor.vertices[simplex,1]
        print ''
        print simplex, vor.vertices[simplex]
        print x1, y1
        ax1.plot(x1, y1, 'k-')


#--------------------------------------------------------------------------
# external ridges
#--------------------------------------------------------------------------
print '-'*47
xc, yc = cp = points.mean(axis=0)           # center point
ax1.plot(xc, yc, '*')

for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.any(simplex < 0):
        print ''
        print 'simplex', simplex, simplex[simplex >= 0], simplex[simplex >= 0][0]
        print 'pointidx', pointidx

        i = simplex[simplex >= 0][0]        # finite end Voronoi vertex index
        x1, y1 = vp = vor.vertices[i]                 # vertex
        x2, y2 = mp = points[pointidx].mean(axis=0)   # middle point

        ax1.plot(x1, y1, 'ks')
        ax1.plot(x2, y2, 'kx')

        # determine external ridge point
        x3, y3 = calc_ext_vertex(vp, mp, cp, bound_xy)


        ax1.plot([x1,x3], [y1,y3], 'k--')

ax1.set_xlim(bx1, bx2)
ax1.set_ylim(by1, by2)
plt.show(True)
