#------------------------------------------------------------------------------
# filename  : nonlinear_oscillation.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.10.12    start
#
#
# Description: 
#   Solve for the motion of the mass
#   Solve a 2nd order ODE using 4th Runge-Kutta method
#
# Reference
#   Computational Physics: Problem Solving with Python, 3rd Edition
#   Rubin H. Landau, Manuel J Páez, Cristian C. Bordeianu 
#------------------------------------------------------------------------------

import numpy as np
import numpy.linalg as la
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.animation import FuncAnimation, ImageMagickWriter




class NonLinearOscillation():
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

        # Variables
        # y0->x, y1->v
        self.y = np.zeros(2)

        # For RK4
        self.k1 = np.zeros(2)
        self.k2 = np.zeros(2)
        self.k3 = np.zeros(2)
        self.k4 = np.zeros(2)





    def fext(self, t, x):
        return 0


    def func(self, t, y):
        m = self.m
        k = self.k
        p = self.p

        x, v = y
        return v, (self.fext(t,x) - k*np.pow(x,p-1))/m


    def rk4(self, t, h):
        y = self.y
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4

        k1 = h*func(t      , y       )
        k2 = h*func(t + h/2, y + k1/2)
        k3 = h*func(t + h/2, y + k2/2)
        k4 = h*func(t + h  , y + k3  )

        y = y + (k1 + 2*k2 + 2*k3 + k4)/6


    def newton_rapson(self):
        maxiter = self.maxiter
        n = self.n
        x = self.x
        f = self.f
        x_history = self.x_history

        eps = 1e-3      # relative tolerance

        for it in range(1,maxiter+1):
            #print('{:3d}'.format(it), flush=True, end='')
            self.func(x, f)
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
            xlim=(-3, lc+3), ylim=(-lc-5, 1),
            xticks=np.linspace(-3,lc+3,lc+7), yticks=np.linspace(-lc-5,1,lc+7), 
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

    tkwds = dict(fontsize=20)
    texts = [ \
            ax.text(0, -5, 'step = 0', **tkwds), \
            ax.text(0, -6, r'$\theta_1 = {:.2f}^o$'.format(np.rad2deg(np.arccos(x[3]))), **tkwds), \
            ax.text(0, -7, r'$\theta_2 = {:.2f}^o$'.format(np.rad2deg(np.arccos(x[4]))), **tkwds), \
            ax.text(0, -8, r'$\theta_3 = {:.2f}^o$'.format(np.rad2deg(np.arccos(x[5]))), **tkwds), \
            ax.text(0, -9,  r'$T_1 = {:.2f} N$'.format(x[6]), **tkwds), \
            ax.text(0, -10, r'$T_2 = {:.2f} N$'.format(x[7]), **tkwds), \
            ax.text(0, -11, r'$T_3 = {:.2f} N$'.format(x[8]), **tkwds)]

    return fig, ax, patches, texts




def animate(i, patches, texts, l1, l2, x_history):
    x = x_history[i,:]

    x1, y1 = l1*x[3], -l1*x[0]
    x2, y2 = x1 + l2*x[4], y1 - l2*x[1]
    patches[1].get_path().vertices[1] = (x1, y1)
    patches[1].get_path().vertices[2] = (x2, y2)
    patches[2].center = (x1, y1)
    patches[3].center = (x2, y2)
    texts[0].set_text('step = {}'.format(i))
    texts[1].set_text(r'$\theta_1 = {:.2f}^o$'.format(np.rad2deg(np.arccos(x[3]))))
    texts[2].set_text(r'$\theta_2 = {:.2f}^o$'.format(np.rad2deg(np.arccos(x[4]))))
    texts[3].set_text(r'$\theta_3 = {:.2f}^o$'.format(np.rad2deg(np.arccos(x[5]))))
    texts[4].set_text(r'$T_1 = {:.2f} N$'.format(x[6]))
    texts[5].set_text(r'$T_2 = {:.2f} N$'.format(x[7]))
    texts[6].set_text(r'$T_3 = {:.2f} N$'.format(x[8]))

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
    fig, ax, patches, texts = init_plot(L, L1, L2, x0)
    ani = FuncAnimation(fig, animate, frames=nframes, 
            fargs=[patches, texts, L1, L2, tmos.x_history], 
            interval=1000, blit=False, repeat=False)
    plt.show()
    #ani.save('two_masses_on_string.png', writer=ImageMagickWriter(fps=1))
    #ani.save('two_masses_on_string.gif', writer=ImageMagickWriter(fps=1))
    #ani.save('two_masses_on_string.mp4', fps=15)
