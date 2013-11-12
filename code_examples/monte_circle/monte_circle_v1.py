#-*- coding: UTF-8 -*-
#--------------------------------------------------------------------------
# Calculate the area of circle using Monte-Carlo integration
# written by Ki-Hwan Kim (kh.kim@kiaps.org)
#
# last update: 2013/10/15
#
# version 1: naive implementation
#--------------------------------------------------------------------------

from __future__ import division
from random import uniform
from time import time



N = input('전체점을 찍을 횟수 N = ')
n = 0   # number of points in a circle
t1 = time()

for i in xrange(N):
    x = uniform(0,1)    # return a random floating point number in the range [0,1)
    y = uniform(0,1)

    if x*x + y*y <= 1:
        n += 1

t2 = time()
Pi = 4*n/N
duration_time = t2 - t1
print '시행횟수 %d 번일 때, Pi = %f' % (N, Pi) 
print '수행시간은 %s sec 입니다.' %(duration_time) 
