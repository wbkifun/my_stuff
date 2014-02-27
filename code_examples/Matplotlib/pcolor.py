from __future__ import division
import numpy
from numpy import exp
import matplotlib.pyplot as plt


elem = [-1, -0.4, 0.4, 1]
ne = 30
ngq = 4

x = numpy.zeros(ne*ngq)
y = numpy.zeros(ne*ngq)
for ei in xrange(ne):
    for j, gq in enumerate(elem):
        i = j + ei*ngq
        x[i] = 1/ne*ei + (gq+1)/(2*ne)
        y[i] = 1/ne*ei + (gq+1)/(2*ne)


X = numpy.zeros((ne*ngq,ne*ngq))
Y = numpy.zeros((ne*ngq,ne*ngq))
C = numpy.ones((ne*ngq,ne*ngq))
for i in xrange(ne*ngq):
    for j in xrange(ne*ngq):
        X[i,j] = x[i]
        Y[i,j] = y[j]

plt.pcolor(X,Y,C)
plt.show()
