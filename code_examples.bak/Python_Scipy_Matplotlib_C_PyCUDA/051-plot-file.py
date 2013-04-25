#!/usr/bin/env python

from pylab import *
from scipy.io.numpyio import fread

Nx, Ny = 400, 350
	
fd = open('./dat/050-A_binary.dat', 'rb')
A = fread( fd, Nx*Ny, 'f' )
A = A.reshape( Nx, Ny )
fd.close()

fd = open('./dat/050-dA_binary.dat', 'rb')
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
