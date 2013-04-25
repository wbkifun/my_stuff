#!/usr/bin/env python

for i in xrange(1, 32):
	nx = i * 32
	n = nx**3
	nbytes = n * 4

	print('i = %d, nx = %d, ' % (i, nx)),
	if nbytes >= 1024**3: 
		print('%1.2f GiB' % (float(nbytes)/(1024**3)))
	elif nbytes >= 1024**2: 
		print('%1.2f MiB' % (float(nbytes)/(1024**2)))
	elif nbytes >= 1024: 
		print('%1.2f KiB' % (float(nbytes)/(1024)))
	else:
		print('%d Bytes' % nbytes)
