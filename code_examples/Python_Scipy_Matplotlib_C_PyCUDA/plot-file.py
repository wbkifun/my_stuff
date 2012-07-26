#!/usr/bin/env python

import sys
from pylab import *
from scipy.io.numpyio import fread

Nx, Ny = 400, 350
	
#path = raw_input('path the *.dat files: ')
path = sys.argv[1]

path1 = path + 'A_binary.dat'
path2 = path + 'dA_binary.dat'

fd = open( path1, 'rb')
A = fread( fd, Nx*Ny, 'f' )
A = A.reshape( Nx, Ny )
fd.close()

fd = open( path2, 'rb')
dA = fread( fd, Nx*Ny, 'f' )
dA = dA.reshape( Nx, Ny )
fd.close()

figure( figsize=(15,5) )
subplot(1,2,1)
imshow( A.T, cmap=cm.hot, origin='lower', interpolation='bilinear' )
title('2D Gaussian')
xlabel('x-axis')
ylabel('y-axis')
colorbar()

subplot(1,2,2)
imshow( dA.T, cmap=cm.hot, origin='lower', interpolation='bilinear' )
title('differential')
xlabel('x-axis')
ylabel('y-axis')
colorbar()

savefig('./png/050.png')
show()
