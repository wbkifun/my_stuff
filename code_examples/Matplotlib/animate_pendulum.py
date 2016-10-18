from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.animation import FuncAnimation
from numpy import sin, cos, sqrt, pi, deg2rad




def init_patches(L, theta0):
    xy0 = (L*sin(theta0), -L*cos(theta0))

    line_verts = [(0,0), xy0]
    line_codes = [Path.MOVETO, Path.LINETO]
    line_path = Path(line_verts, line_codes)

    kwds = dict(ec='k', alpha=1)
    patches = [ \
            ax.add_patch(PathPatch(line_path, lw=2, fc='k', fill=False, **kwds)), \
            ax.add_patch(Circle(xy=xy0, radius=radius, fc='r', **kwds)), \
            ax.add_patch(Rectangle(xy=(-.5, 0), width=1, height=0.1, hatch='/', fc='g', **kwds)) ]

    return patches




def animate(i, nframes, patches, G, L, theta0):
    t = i/(nframes-1)*(2*pi*sqrt(G/L))
    theta = theta0*sin(sqrt(G/L)*t)
    xy = (L*sin(theta), -L*cos(theta))
    patches[0].get_path().vertices[-1] = xy
    patches[1].center = xy

    return patches
    



if __name__ == '__main__':
    #
    # Define the pendulum
    #
    G = 9.8             # gravity acceleration [m/s^2]
    radius = 0.05       # radius of sphere [m]
    L = 0.8 - radius    # length of pendulum [m]
    theta0 = deg2rad(20)    # initial angle [rad]

    #
    # Set up the axes, making sure the axis ratio is equal
    #
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], # l,b,w,h
            xlim=(-0.5, 0.5), ylim=(-0.9, 0.1),
            xticks=np.linspace(-.5,.5,11), yticks=np.linspace(-.9,.1,11), 
            aspect='equal', frameon=True)
    ax.grid(True)

    #
    # Animation
    #
    patches = init_patches(L, theta0)
    nframes = 2400
    ani = FuncAnimation(fig, animate, frames=nframes, 
            fargs=[nframes, patches, G, L, theta0], 
            interval=50, blit=False, repeat=False)
    plt.show()
    #ani.save('pendulum.mp4', fps=15)
