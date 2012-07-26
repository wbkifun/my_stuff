#!/usr/bin/env python

import sys
import numpy as np

import fib1
print fib1.fib.__doc__
a = np.zeros(20, 'd')
fib1.fib(a)
print a


import fib2
print fib2.fib.__doc__
b = fib2.fib(10)
print b, b.dtype, b.shape


import fib3
print fib3.fib.__doc__
c = np.zeros_like(a)
fib3.fib(c)
print c, c.dtype, c.shape


import fib4
print fib4.fib.__doc__
d = np.zeros_like(a)
fib4.fib(d)
print d, d.dtype, d.shape
