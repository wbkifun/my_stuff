from __future__ import division
import numpy
from numpy.random import rand
from numpy import sqrt, fabs, sign
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def get_triangle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    '''
    # regular triangle
    x3 = 0.5*(x1+x2+sqrt(3)*sign(x1-x2)*(y1-y2))
    y3 = 0.5*(y1+y2-sqrt(3)*fabs(x1-x2))

    x4 = 0.5*(x1+x2-sqrt(3)*sign(x1-x2)*(y1-y2))
    y4 = 0.5*(y1+y2+sqrt(3)*fabs(x1-x2))

    x5 = 0.5*(x1+x2-sqrt(3)*(y1-y2))
    y5 = 0.5*(y1+y2+sqrt(3)*(x1-x2))
    '''


    # isosceles triangle
    x3 = 0.5*(x1+x2+sign(x1-x2)*(y1-y2))
    y3 = 0.5*(y1+y2-fabs(x1-x2))

    x4 = 0.5*(x1+x2-sign(x1-x2)*(y1-y2))
    y4 = 0.5*(y1+y2+fabs(x1-x2))

    x5 = 0.5*(x1+x2-(y1-y2))
    y5 = 0.5*(y1+y2+(x1-x2))


    return (x3,y3), (x4,y4), (x5,y5)




fig = plt.figure(figsize=(12,16))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)


set1 = (-1,-1), (1,1)
set2 = (1,-1), (-1,1)
set3 = (1,1), (-1,-1)
set4 = (-1,1), (1,-1)
set5 = (0,-1), (0,1)
set6 = (-1,0), (1,0)


ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]
xy_set_list = [set1,set2,set3,set4,set5,set6]

for ax, (pt1, pt2) in zip(ax_list, xy_set_list):
    (x1,y1), (x2,y2) = pt1, pt2
    (x3,y3), (x4,y4), (x5,y5) = get_triangle(pt1, pt2)

    # regular triangle
    #d = sqrt((x2-x1)**2 + (y2-y1)**2)

    # isosceles triangle
    d = sqrt((x2-x1)**2 + (y2-y1)**2)/sqrt(2)

    ax.add_patch( Circle((x1,y1), d, fill=False, ec='k') )
    ax.add_patch( Circle((x2,y2), d, fill=False, ec='k') )

    ax.plot(x1, y1, 'or')
    ax.plot(x2, y2, 'ob')
    ax.plot(x5, y5, 'pc', ms=15)
    ax.plot(x3, y3, 'dc')
    ax.plot(x4, y4, 'sc')

plt.show(True)
