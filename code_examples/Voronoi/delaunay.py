from __future__ import division
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt



#----------------------------------------------------
# Random points
#----------------------------------------------------
N = 10
#xys = np.random.rand(N,2)
#np.save('random%d.npy'%N, xys)
xys = np.load('random%d.npy'%N)




#----------------------------------------------------
# Delaunay triangles
#----------------------------------------------------
tri = Delaunay(xys)

print '\nsimplices'
for i, simplex in enumerate(tri.simplices):
    print i, simplex

print '\nneighbors'
for i, neighbor in enumerate(tri.neighbors):
    print i, neighbor

print '\nvertex_to_simplex'
print tri.vertex_to_simplex



#----------------------------------------------------
# Plot
#----------------------------------------------------
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1)
xs, ys = xys[:,0], xys[:,1]


# plot points
ax.scatter(xs, ys)
for i in range(N):
    ax.annotate(i, (xs[i],ys[i]), size=18)


# plot triangles
plt.triplot(xys[:,0], xys[:,1], tri.simplices)
for i, simplex in enumerate(tri.simplices):
    tri_mid = xys[simplex].mean(axis=0)
    ax.annotate(i, tri_mid, size=10, bbox=dict(boxstyle='round',fc='0.8'))


plt.show(True)
