from __future__ import division
import numpy


nn = 10
dx = 0.12
dy = 0.13

f = open('write_ascii.dat', 'w')
f.write('%d\n'%nn)
for i in xrange(nn):
    for j in xrange(nn):
        x = i*dx
        y = j*dy
        f.write('%f\t%f\t%f\n' % (x, y, 2*x+y))

f.close()
