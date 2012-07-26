#!/usr/bin/env python
#  Filename:  mandelbrot.py
#  For  PU  at  MNG  Raemibuehl,  by  Anh  Huy  Truong 

import numpy as np
import matplotlib.pyplot as plt
def mandelbrot(steps, start, stepx, stepy, iterations):
	c = start
	c0 = start
	res = []
	for h in xrange(steps):
		for k in xrange(steps):
			if abs(c) >= 2: # check divergence
				start  +=  stepx
			else:
				for i in xrange(iterations):
					c = c*c + c0

					if abs(c) >= 2:		# check divergence
						break
				else:
					res.append(c0)	# add point to list
				start += stepx
			c = start
			c0 = start
		start = start + stepy - complex(steps * stepx.real,0)
	return res

if __name__ == "__main__":
	steps = 1201
	start = complex(-3,-3)
	stepx = complex(0.005,0)
	stepy = complex(0,0.005)
	iters = 100
	res = mandelbrot(steps, start, stepx, stepy, iters)
	res = np.array(res)
	plt.plot(res.real, res.imag, ',k')
	plt.xlabel(r'$re(z)$')
	plt.ylabel(r'$i \cdot im(z)$')
	plt.title('The Mandelbrot Set')
	plt.axis([-3,2,-2,2])
	plt.grid(True)
	plt.show()
