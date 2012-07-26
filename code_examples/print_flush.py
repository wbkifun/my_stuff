#!/usr/bin/env python

import sys, time

tmax = 100
for i in range(tmax):
	print "[tstep=\t%d/%d (%d %s)]\r" % (i, tmax, i/100.*100, '%'),
	sys.stdout.flush()
	time.sleep(1)
