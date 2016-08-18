#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt



nx, ny = 150, 100
x1, x2 = -15, 15
y1, y2 = -10, 10

x = np.linspace(x1, x2, nx)
y = np.linspace(y1, y2, ny)

xx, yy = np.meshgrid(x, y)          # 2d index array
f = np.exp( -(xx**2 + yy**2)/10 )   # Gaussian function


# check the initial state
plt.ion()
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Moving 2D Gaussian')
#ax.set_xticks([-15, -12.5, -5.3, 0, 4.2, 8.6, 15])

#img = ax.imshow(f.T, origin='lower', aspect='auto')
img = ax.imshow(f.T, origin='lower', extent=[x1,x2,y1,y2], aspect='auto')

fig.colorbar(img)
plt.show(True)              # show the static plot


# animate
'''
dx = x[1] - x[0]
dy = y[1] - y[0]
for tstep in range(50):
    f[:,:] = np.exp( -((xx-tstep*dx)**2 + (yy-tstep*dy)**2)/10 )

    img.set_data(f)               # update the line object
    plt.draw(True)                     # redraw
'''
