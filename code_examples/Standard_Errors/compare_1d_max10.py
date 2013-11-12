from __future__ import division
import numpy
from numpy import sqrt, exp
from numpy.random import randn
from math import fsum
import matplotlib.pyplot as plt


from standard_errors import ste_1_2_inf


nn = 1000
sigma = 0.4

x = numpy.linspace(-2, 2, nn)
f0 = 10*exp( -x**2/(2*sigma**2) )
f1 = f0*(1 - randn(nn)/100)
f2 = f0*(1 - randn(nn)/50)
f3 = f0*(1 - randn(nn)/25)
f4 = f0*(1 - randn(nn)/10)

fig = plt.figure(figsize=(15,12))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot(x, f1, 'r')
ax1.plot(x, f0, 'k')
ax1.set_ylim(-1, 12)
ax1.set_title('(a) error amplitude $1/100$')

ax2.plot(x, f2, 'r')
ax2.plot(x, f0, 'k')
ax2.set_ylim(-1, 12)
ax2.set_title('(b) error amplitude $1/50$')

ax3.plot(x, f3, 'r')
ax3.plot(x, f0, 'k')
ax3.set_ylim(-1, 12)
ax3.set_title('(c) error amplitude $1/25$')

ax4.plot(x, f4, 'r')
ax4.plot(x, f0, 'k')
ax4.set_ylim(-1, 12)
ax4.set_title('(d) error amplitude $1/10$')



print 'nn=', nn
print 'L1, L2, Linf\n'

for i in xrange(4):
    fx = [f1, f2, f3, f4][i]
    L1, L2, Linf = ste_1_2_inf(f0, fx)

    print 'f%d\t(%1.5f, %1.5f, %1.5f)' % (i+1, L1, L2, Linf)


plt.show()
