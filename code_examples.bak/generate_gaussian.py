from __future__ import division
import numpy
import matplotlib.pyplot as plt


mean = [0,0]
cov = [[1,0], [0,10]]      # diagonal covariance, points lie on x or y-axis
x, y = numpy.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x,y,'x')
plt.axis('equal')
plt.show()
