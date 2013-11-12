from __future__ import division
import numpy
from numpy import sqrt, exp
from numpy.random import randn
from math import fsum
import matplotlib.pyplot as plt


from standard_errors import ste_1_2_inf


nn = 100
sigma = 0.6

x = numpy.linspace(-2, 2, nn)
y = numpy.linspace(-2, 2, nn)
xx, yy = numpy.meshgrid(x, y, sparse=True)

f0 = exp( -(xx**2 + yy**2)/(2*sigma**2) )
f1 = f0*(1 - randn(nn,nn)/100)
f2 = f0*(1 - randn(nn,nn)/50)
f3 = f0*(1 - randn(nn,nn)/25)
f4 = f0*(1 - randn(nn,nn)/10)

fig = plt.figure(figsize=(15,12))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.contour(x,y,f1)
ax1.imshow(f1, cmap='OrRd', extent=(-2,2,-2,2))
ax1.set_title('(a) error amplitude $1/100$')

ax2.contour(x,y,f2)
ax2.imshow(f2, cmap='OrRd', extent=(-2,2,-2,2))
ax2.set_title('(b) error amplitude $1/50$')

ax3.contour(x,y,f3)
ax3.imshow(f3, cmap='OrRd', extent=(-2,2,-2,2))
ax3.set_title('(c) error amplitude $1/25$')

ax4.contour(x,y,f4)
ax4.imshow(f4, cmap='OrRd', extent=(-2,2,-2,2))
ax4.set_title('(d) error amplitude $1/10$')


print 'nn=', nn
print 'L1, L2, Linf\n'

for i in xrange(4):
    fx = [f1, f2, f3, f4][i]
    L1, L2, Linf = ste_1_2_inf(f0, fx)

    print 'f%d\t(%1.5f, %1.5f, %1.5f)' % (i+1, L1, L2, Linf)


plt.show()
