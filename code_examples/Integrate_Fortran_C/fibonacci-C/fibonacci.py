#!/usr/bin/env python

import numpy as np

import fib1
print fib1.fib.__doc__
a = np.zeros(20, 'd')
fib1.fib(a)
print a
