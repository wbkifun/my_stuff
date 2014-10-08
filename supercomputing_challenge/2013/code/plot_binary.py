from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys



class PlotSlit(object):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

        plt.ion()
        fig = plt.figure(figsize=(12,8))
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

        field[:] = 0
        img = ax.imshow(field.T, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(img)


    def plot_single_slit(self, x0, width, depth):
        nx, ny = self.nx, self.ny
        ax = self.ax

        box1 = plt.Rectangle((x0, 0), depth, ny/2-width/2, fc='k')
        box2 = plt.Rectangle((x0, ny/2+width/2), depth, ny/2-width/2, fc='k')
        ax.add_patch(box1)
        ax.add_patch(box2)


    def plot_double_slit(self, x0, width, depth, distance):
        nx, ny = self.nx, self.ny
        ax = self.ax

        box1 = plt.Rectangle((x0, 0), depth, ny/2-(distance+width)/2, fc='k')
        box2 = plt.Rectangle((x0, ny/2-(distance-width)/2), depth, (distance-width), fc='k')
        box3 = plt.Rectangle((x0, ny/2+(distance+width)/2), depth, ny/2-(distance+width)/2, fc='k')
        ax.add_patch(box1)
        ax.add_patch(box2)
        ax.add_patch(box3)




if __name__ == '__main__':
    # setup
    nx, ny = 1200, 1000

    pst = PlotSlit(nx, ny)
    pst.read_binary_file()

    pst.plot_single_slit(x0=2*nx/3, width=10, depth=5)
    #pst.plot_double_slit(x0=2*nx/3, width=10, depth=5, distance=100)

    #pst.plot_field(vmin=-0.1, vmax=0.1)     # for circular wave
    #pst.plot_field()                        # for line wave
    pst.plot_field(vmin=-0.2, vmax=0.2)     # for slit

    plt.show(True)
