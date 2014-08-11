from __future__ import division
import numpy as np
from numpy import sqrt
from shapely.geometry import Point, LineString, LinearRing


pt = Point(0,0)
print 'pt', pt
print 'pt type', pt.geom_type
print 'pt coordinate:', pt.coords[:][0]


#line = LineString([(0,1),(-1,0)])
line = LineString([(-1,-1),(1,1)])
print 'line', line
print 'line type', line.geom_type
print 'line coordinates:', line.coords[:]
print 'line length:', line.length, '== sqrt(2)*2', line.length==sqrt(2)*2


print ''
d = pt.distance(line)
print 'distance:', d

ipt = pt.intersection(line)
print 'ipt type', ipt.geom_type
print ipt


print ''
lr = LinearRing(line.coords[:]+pt.coords[:])
print 'is_ccw', lr.is_ccw
