#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

x=.1
m=np.arange(1., 4., 0.01)
y=[]
error=1.
yarray=[]
ymin, ymax = 0, 0

def iteration(x, mu):
	return mu*x*(1-x)

for i in xrange(len(m)):
	x = .1
	for j in xrange(10000):
		x = iteration(x, m[i])
	
	while(error>1.e-3):
		y.append(x)
		x1 = iteration(x, m[i])
		yy=np.array(y)
		error=min(abs(yy-x1)/(yy+x1)*2)
		x=x1
		
	if max(y) > ymax: ymax=max(y)
	if min(y) < ymin: ymin=min(y)
	yarray.append(y)	
	y=[]
	error = 1.
	#print float(i)/float(len(m)) *100,"%"
	
print ymax, ymin
my = np.arange(ymin, ymax, 0.01)
plane = np.zeros((m.size,my.size), 'i')
for i, y_list in enumerate(yarray):
	for y in y_list:
		plane[i,abs(my-y).argmin()]=1

plt.imshow(plane.T, origin='lower')
plt.xticks(range(m.size),m)
plt.colorbar()
plt.show()
