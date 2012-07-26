#!/usr/bin/env python

import numpy as np
import h5py as h5


print '[From h5 file]'

# save to the h5 file
f = h5.File('h5_data.h5', 'r')
nx = f.attrs['nx']
ny = f.attrs['ny']
dx = f.attrs['dx']
dy = f.attrs['dy']
xticks = f['ticks']['xticks'].value
yticks = f['ticks']['yticks'].value
psir = f['data']['psir'].value
psic = f['data']['psic'].value

# spec
print 'nx, ny:', nx, ny
print 'dx, dy:', dx, dy

# ticks (1d array)
print 'xticks:', xticks
print 'yticks:', yticks

# data (2d array)
print 'psir:'
print psir
print 'psic:'
print psic

