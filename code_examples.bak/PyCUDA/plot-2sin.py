#!/usr/bin/env python

from pylab import *

x = arange(0,100)

y1 = sin(2*pi*x/100)
y2 = cos(2*pi*x/100)

plot(x,y1,linewidth=2)
hold(True)
plot(x,y2,linewidth=2)
xlabel('x'); ylabel('y')
title('Plotting sin(x) & cos(x)')
legend(['sin(x)','cos(x)'])
grid(True)

show()
