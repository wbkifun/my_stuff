import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection


#fig, ax = plt.subplots()
fig = plt.figure(figsize=(12,12))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
ax = fig.add_subplot(1,1,1)


# Circle
center, radius = (0.4,0.3), 0.1
ax.add_patch( Circle(center, radius, fill=False, hatch='/') )


# Polygon
points = [(0.3,0.2), (0.4,0.3), (0.38,0.4), (0.25,0.45), (0.2,0.35)]
#points = [(0,0)]
polygon = Polygon(points, fill=True, color='b', alpha=0.4)
ax.add_patch(polygon)
#pc = PatchCollection( [polygon] )
#ax.add_collection(pc)


# animation
def ani_func(seq, polygon, pts):
    new_pts = np.array(pts) + seq*0.02
    polygon.set_xy(new_pts)
    return polygon, 

ani = FuncAnimation(fig, ani_func, frames=20, fargs=(polygon,points), interval=20, blit=True)

"""
#----------------------------------------------------------------------------
# Polygons
# !! not work 2014.3.15
#----------------------------------------------------------------------------
points1 = [(0.3,0.2), (0.4,0.3), (0.38,0.4), (0.25,0.45), (0.2,0.35)]
points2 = [(0,0), (1,0), (1,1), (0,1)]
polygon1 = Polygon(points1, fill=True, color='b', alpha=0.4)
polygon2 = Polygon(points2, fill=True, color='y', alpha=0.4)
polygons = [polygon1, polygon2]
pc = PatchCollection(polygons, facecolors='w', edgecolors='k')
ax.add_collection(pc)

# animation
def ani_func(seq, pc, polygons):
    for polygon in polygons:
        xy = polygon.get_xy()
        new_xy = xy + seq*0.02
        polygon.set_xy(new_xy)
    return pc, 

ani = FuncAnimation(fig, ani_func, frames=20, fargs=(pc, polygons), interval=20, blit=True)
"""

plt.show()
