from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys



class PlotWave2D(object):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        plt.ion()
        fig = plt.figure(figsize=(12,12))
        self.ax = fig.add_subplot(1,1,1)


    def read_binary_file(self):
        nx, ny = self.nx, self.ny
        try:
            fpath = sys.argv[1]
            fp = open(fpath, 'rb')

        except IndexError:
            print 'Usage:'
            print '$ python plot_binary.py binary_file_name'
            sys.exit()

        self.field = np.fromfile(fp, count=nx*ny, dtype=np.float64).reshape((nx,ny), order='F')


    def plot_field(self, vmin=None, vmax=None):
        ax = self.ax
        field = self.field

        img = ax.imshow(field.T, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(img)


    def plot_single_slit(self, x0, width, depth):
        nx, ny = self.nx, self.ny
        ax = self.ax

        box1 = plt.Rectangle((x0, 0), depth, ny/2-width/2, fc='k')
        box2 = plt.Rectangle((x0, ny/2+width/2), depth, ny/2-width/2, fc='k')
        ax.add_patch(box1)
        ax.add_patch(box2)


    def plot_double_slit(self, x0, width, depth, gap):
        nx, ny = self.nx, self.ny
        ax = self.ax

        box1 = plt.Rectangle((x0, 0), depth, ny/2-(gap+width)/2, fc='k')
        box2 = plt.Rectangle((x0, ny/2-(gap-width)/2), depth, (gap-width), fc='k')
        box3 = plt.Rectangle((x0, ny/2+(gap+width)/2), depth, ny/2-(gap+width)/2, fc='k')
        ax.add_patch(box1)
        ax.add_patch(box2)
        ax.add_patch(box3)




if __name__ == '__main__':
    # setup
    #nx, ny = 1000, 1000
    nx, ny = 2000, 2000
    #nx, ny = 4000, 4000
    #nx, ny = 6000, 6000
    #nx, ny = 8000, 8000
    #nx, ny = 16000, 16000

    pw = PlotWave2D(nx, ny)
    pw.read_binary_file()

    #pw.plot_single_slit(x0=2*nx/3, width=10, depth=5)
    #pw.plot_double_slit(x0=2*nx/3, width=10, depth=5, gap=100)
    #pw.plot_double_slit(x0=nx/2, width=40, depth=20, gap=400)
    #pw.plot_double_slit(x0=nx/2, width=100, depth=50, gap=1000)

    #pw.plot_field(vmin=-0.1, vmax=0.1)     # for circular wave
    #pw.plot_field()                        # for line wave
    pw.plot_field(vmin=-0.1, vmax=0.1)     # for slit

    plt.show(True)
