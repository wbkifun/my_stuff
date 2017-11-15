#------------------------------------------------------------------------------
# filename  : two_masses_on_string_1.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.10.12    from original version
#                           Refactoring using class
#                           Animation using Matplotlib
#
#
# Description: 
#   Find angles and tensions of two masses on a string
#   Solve a physics problem using Newton-Rapson method and Matrix solver
#
# Reference
#   Computational Physics: Problem Solving with Python, 3rd Edition
#   Rubin H. Landau, Manuel J Páez, Cristian C. Bordeianu 
#------------------------------------------------------------------------------

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.animation import FuncAnimation




class TwoMassesOnString():
    def __init__(self, lc, l1, l2, l3, w1, w2, x0, maxiter=100):
        self.lc = lc    # length of ceiling [m]
        self.l1 = l1    # length of string [m]
        self.l2 = l2
        self.l3 = l3
        self.w1 = w1    # weight of mass [N]
        self.w2 = w2
        self.x0 = x0
        self.maxiter = maxiter

        self.n = 9      # number of variables

        # x = [sin(theta1), sin(theta2), sin(theta3), 
        #      cos(theta1), cos(theta2), cos(theta3), 
        #      T1, T2, T3]
        self.x = np.zeros(n)
        self.f = np.zeros(n)
        self.df = np.zeros((n,n))

        # Initialization
        self.x[:] = x0

        # For animation
        self.x_history = np.zeros((maxiter+1, n))
        self.x_history[0,:] = self.x[:]



    def func(self):
        lc, l1, l2, l3 = self.lc, self.l1, self.l2, self.l3
        w1, w2 = self.w1, self.w2
        x = self.x
        f = self.f

        f[0] = l1*x[3] + l2*x[4] + l3*x[5] - lc
        f[1] = l1*x[0] + l2*x[1] - l3*x[2]
        f[2] = x[6]*x[0] - x[7]*x[1] - w1
        f[3] = x[6]*x[3] - x[7]*x[4]
        f[4] = x[7]*x[1] + x[8]*x[2] - w2
        f[5] = x[7]*x[4] - x[8]*x[5]
        f[6] = x[0]**2 + x[3]**2 - 1.0
        f[7] = x[1]**2 + x[4]**2 - 1.0
        f[8] = x[2]**2 + x[5]**2 - 1.0


    def dfdx(self):
        n = self.n
        x = self.x
        f = self.f
        df = self.df

        h = 1e-4

        for j in range(n):
            tmp = x[j]
            x[j] += h/2
            self.func()
            for i in range(n): df[i,j] = f[i]
            x[j] = tmp

        for j in range(n):
            tmp = x[j]
            x[j] -= h/2
            self.func()
            for i in range(n): df[i,j] = (df[i,j] - f[i])/h
            x[j] = tmp


    def newton_rapson(self):
        maxiter = self.maxiter
        n = self.n
        x = self.x
        f = self.f
        x_history = self.x_history

        eps = 1e-3      # relative tolerance

        for it in range(1,maxiter+1):
            #print('{:3d}'.format(it), flush=True, end='')
            self.func()
            self.dfdx()
            dx = la.solve(self.df, -self.f)
            x[:] += dx[:]
            x_history[it,:] = x[:]

            errX = errF = errXi = 0.0
            for i in range(n):
                if x[i] != 0: errXi = abs(dx[i]/x[i])
                else:         errXi = abs(dx[i])
                if errXi > errX: errX = errXi
                if abs(f[i]) > errF: errF = abs(f[i])
            if (errX <= eps) and (errF <= eps): break

        return it




def init_plot(lc, l1, l2, x):
    #
    # Figure and Axis
    #
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], # l,b,w,h
            xlim=(-1, lc+1), ylim=(-lc, 1),
            xticks=np.linspace(-1,lc+1,lc+3), yticks=np.linspace(-lc-1,1,lc+3), 
            aspect='equal', frameon=True)
    ax.grid(True)

    #
    # Patches
    #
    x1, y1 = l1*x[3], -l1*x[0]
    x2, y2 = x1 + l2*x[4], y1 - l2*x[1]

    line_verts = [(0,0), (x1,y1), (x2,y2), (lc,0)]
    line_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    line_path = Path(line_verts, line_codes)

    kwds = dict(ec='k', alpha=1)
    patches = [ \
            ax.add_patch(Rectangle(xy=(-1, 0), width=lc+2, height=1, hatch='/', fc='gray', **kwds)), \
            ax.add_patch(PathPatch(line_path, lw=2, fc='black', fill=False, **kwds)), \
            ax.add_patch(Circle(xy=(x1,y1), radius=0.3, fc='cyan', **kwds)), \
            ax.add_patch(Circle(xy=(x2,y2), radius=0.6, fc='yellow', **kwds))]

    return fig, ax, patches




def animate(i, patches, l1, l2, x_history):
    x = x_history[i,:]

    x1, y1 = l1*x[3], -l1*x[0]
    x2, y2 = x1 + l2*x[4], y1 - l2*x[1]
    patches[1].get_path().vertices[1] = (x1, y1)
    patches[1].get_path().vertices[2] = (x2, y2)
    patches[2].center = (x1, y1)
    patches[3].center = (x2, y2)

    return patches
    



if __name__ == '__main__':
    #
    # Setup
    #
    n = 9
    L, L1, L2, L3 = 8, 3, 4, 4
    W1, W2 = 10, 20

    x0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1.]
    tmos = TwoMassesOnString(L, L1, L2, L3, W1, W2, x0)
    num_iter = tmos.newton_rapson()

    print('Number of iterations =', num_iter)
    print('Solution =\n', tmos.x)

    #
    # Animation using Matplotlib
    #
    nframes = num_iter + 1
    fig, ax, patches = init_plot(L, L1, L2, x0)
    ani = FuncAnimation(fig, animate, frames=nframes, 
            fargs=[patches, L1, L2, tmos.x_history], 
            interval=500, blit=False, repeat=False)
    plt.show()
    #ani.save('two_masses_on_string.mp4', fps=15)
