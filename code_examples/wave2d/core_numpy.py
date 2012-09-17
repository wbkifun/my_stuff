from __future__ import division


si = slice(1,-1)
def advance(c, f, g):
	f[si,si] = c[si,si]*(g[2:,si] + g[:-2,si] + g[si,2:] + g[si,:-2] - 4*g[si,si]) \
            + 2*g[si,si] - f[si,si]
