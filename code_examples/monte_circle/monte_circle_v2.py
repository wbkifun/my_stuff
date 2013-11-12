#-*- coding: UTF-8 -*-
#--------------------------------------------------------------------------
# Calculate the area of circle using Monte-Carlo integration
# written by Ki-Hwan Kim (kh.kim@kiaps.org)
#
# last update: 2013/10/15
#
# version 1: naive implementation
# version 2: using numpy
#--------------------------------------------------------------------------

from __future__ import division
import numpy
from time import time


N = input('전체점을 찍을 횟수 N = ')
t1 = time()

# N-size array of random floating point number in the range [0,1)
x_arr = numpy.random.uniform(0,1,N)     
y_arr = numpy.random.uniform(0,1,N)     

# count the number of points in the circle
r_arr = x_arr**2 + y_arr**2
n = numpy.count_nonzero(r_arr <= 1)
Pi = 4*n/N

t2 = time()
duration_time = t2 - t1
print '시행횟수 %d 번일 때, Pi = %f' % (N, Pi) 
print '수행시간은 %s sec 입니다.' %(duration_time) 
