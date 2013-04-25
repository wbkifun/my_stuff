#!/usr/bin/env python

import sys
import os
os.system("f2py -m hw_f -c --fcompiler=gnu95 ./Fortran/hw.f90 > /dev/null")
from hw_f import hw1, hw2

try:
	a, b = float(sys.argv[1]), float(sys.argv[2])
except IndexError:
	print 'Usage:', sys.argv[0], 'a b'
	sys.exit(1)

print 'hw1, result:', hw1(a, b)
print 'hw2, result:', 
hw2(a, b)
