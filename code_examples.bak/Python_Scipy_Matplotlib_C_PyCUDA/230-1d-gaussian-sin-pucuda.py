#!/usr/bin/env python

import scipy
import matplotlib.pyplot as pyplot
import pycuda.autoinit
from pycuda import gpuarray

x = scipy.arange(-5,5,0.01,'f')
a = scipy.exp(-(x**2)/2)
b = scipy.sin(5*x)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c = (a_gpu*b_gpu).get()

pyplot.plot(x, a, x, b)
pyplot.plot(x, c, linewidth=3)
pyplot.savefig('./png100/1d-gaussian-sin-gpuarray.png')
pyplot.show()
