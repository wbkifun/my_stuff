from __future__ import division
import numpy as np
from numpy import sqrt
from shapely.geometry import Point, LineString


line1 = LineString([(-1,0),(0,1)])
print 'line1', line1
print 'line1 type', line1.geom_type
print 'line1 coordinates:', line1.coords[:]
print 'line1 length:', line1.length, '== sqrt(2)', line1.length==sqrt(2)


print ''
line2 = LineString([(0,0),(-0.5,0.5)])
print 'line2', line2
print 'line2 type', line2.geom_type
print 'line2 coordinates:', line2.coords[:]
print 'line2 length:', line2.length, '== sqrt(2)/2', line2.length==sqrt(2)/2


print ''
line3 = LineString([(0,0),(-1,1)])
print 'line3', line3
print 'line3 type', line3.geom_type
print 'line3 coordinates:', line3.coords[:]
print 'line3 length:', line3.length, '== sqrt(2)', line3.length==sqrt(2)



print ''
print 'line1-line2'
ipt = line1.intersection(line2)
print 'ipt', ipt
print 'touches:', line1.touches(line2)
print 'crosses:', line1.crosses(line2)



print ''
print 'line1-line3'
ipt = line1.intersection(line3)
print 'ipt', ipt
print 'touches:', line1.touches(line3)
print 'crosses:', line1.crosses(line3)
