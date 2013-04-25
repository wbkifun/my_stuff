#!/usr/bin/env python

import numpy as np
import h5py as h5


print '[Original]'

# spec
nx, ny = 7, 9
dx = 0.01
dy = 0.0005
print 'nx, ny:', nx, ny
print 'dx, dy:', dx, dy

# ticks (1d array)
xticks = np.arange(nx) * dx
yticks = np.arange(ny) * dy
print 'xticks:', xticks
print 'yticks:', yticks

# data (2d array)
psir = np.ones((nx, ny), dtype=np.float32)
psic = np.ones((nx, ny), dtype=np.complex64)
print 'psir:'
print psir
print 'psic:'
print psic

# write to the h5 file
f = h5.File('h5_data.h5', 'w')
f.attrs['nx'] = nx
f.attrs['ny'] = ny
f.attrs['dx'] = dx
f.attrs['dy'] = dy
f.create_group('ticks')
f['ticks'].create_dataset('xticks', data=xticks, compression='gzip')
f['ticks'].create_dataset('yticks', data=yticks, compression='gzip')
f.create_group('data')
f['data'].create_dataset('psir', data=psir, compression='gzip')
f['data'].create_dataset('psic', data=psic, compression='gzip')
f.close()
