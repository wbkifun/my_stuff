from __future__ import division
import numpy
from numpy import pi, arctan, sin, cos, abs
import matplotlib.pyplot as plt
import time




#-----------------------------------------
# convet angle [alpha,beta] -> longitude
#-----------------------------------------
N = 10
alphas = numpy.linspace(-pi/4, pi/4, N+1)
betas = numpy.linspace(-pi/4, pi/4, N+1)

lons = numpy.zeros((alphas.size, betas.size), order='F')
for j, beta in enumerate(betas):
    for i, alpha in enumerate(alphas):
        if abs(beta) < 1e-10:
            if alpha < 0: lon = 3*pi/2
            else: lon = pi/2
        else:
            lon = -arctan(alpha/beta)
            if beta > 0: lon += pi
            elif alpha < 0: lon += 2*pi

        lons[i,j] = lon



#-----------------------------------------
# plot
#-----------------------------------------
def get_arc_pts(angle, radius=0.1, numpts=100):
    '''
    angle from -y axis(-pi/2) on the xy plane
    '''

    angles = numpy.linspace(0, angle, numpts)
    xs = radius * sin(angles)
    ys = - radius * cos(angles)

    return xs, ys



plt.ion()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)

ax.plot([-pi/4,pi/4], [0,0], '-k')  # x-axis
ax.plot([0,0], [-pi/4,pi/4], '-k')  # y-axis
ax.set_xlim(-pi/4, pi/4)
ax.set_ylim(-pi/4, pi/4)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.set_title(r'Convert angles $[\alpha, \beta] \rightarrow \lambda$ in Face 6')

pt, = ax.plot(0, 0, 'o')
line, = ax.plot([0,0], [0,0], '-', linewidth=2)
xs, ys = get_arc_pts(0)
arc, = ax.plot(xs, ys, '-')

for j, beta in enumerate(betas):
    for i, alpha in enumerate(alphas):
        lon = lons[i,j]

        pt.set_data(alpha, beta)
        line.set_data([0,2*sin(lon)], [0,-2*cos(lon)])

        xs, ys = get_arc_pts(lon)
        arc.set_data(xs, ys)

        plt.draw()
        time.sleep(0.1)


plt.show(True)
