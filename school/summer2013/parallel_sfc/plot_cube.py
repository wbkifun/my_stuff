from __future__ import division
import numpy
import matplotlib.pyplot as plt



def plot_cube(cube_seq, cube_domain):
    N = cube_seq.shape[0]

    # figure
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(1,1,1)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    # draw the sequence numbers
    for face in xrange(6):
        for j in xrange(N):
            for i in xrange(N):
                gi = {0:N, 1:2*N, 2:N, 3:0, 4:N, 5:3*N}[face] + i
                gj = {0:N, 1:N, 2:2*N, 3:N, 4:0, 5:N}[face] + j
                ax.text(gi, gj, cube_seq[i,j,face], \
                        fontsize=4, \
                        horizontalalignment='center', \
                        verticalalignment='center')

    # draw the lines
    for i in xrange(N+1):
        ax.plot([-0.5,4*N+0.5], [N-0.5+i,N-0.5+i], 'k-')
        ax.plot([N-0.5+i,N-0.5+i], [-0.5,3*N+0.5], 'k-')

    for i in xrange(N):
        ax.plot([N-0.5,2*N-0.5], [0.5+i,0.5+i], 'k-')
        ax.plot([N-0.5,2*N-0.5], [2*N+0.5+i,2*N+0.5+i], 'k-')
        ax.plot([0.5+i,0.5+i], [N-0.5,2*N-0.5], 'k-')
        ax.plot([2*N+0.5+i,2*N+0.5+i], [N-0.5,2*N-0.5], 'k-')
        ax.plot([3*N+0.5+i,3*N+0.5+i], [N-0.5,2*N-0.5], 'k-')

    # draw the partitions
    domain = numpy.zeros((4*N,3*N), 'f4')
    for face in xrange(6):
        curve = cube_domain[:,:,face]
        i, j = [(1,1), (2,1), (1,2), (0,1), (1,0), (3,1)][face]
        domain[i*N:(i+1)*N,j*N:(j+1)*N] = curve

    ax.imshow(domain.T, cmap='Paired', origin='lower', interpolation='nearest')

    #plt.savefig('cube.png', dpi=300)
    #plt.savefig('cube.eps', dpi=300)
    plt.show()




if __name__ == '__main__':
    #--------------------------------------------------------------------------
    # cube_proc
    from space_filling_curve import SpaceFillingCurve

    cube = SpaceFillingCurve(N=3, ngq=4, nproc=2)
    plot_cube(cube.cube_proc, cube.cube_proc)
