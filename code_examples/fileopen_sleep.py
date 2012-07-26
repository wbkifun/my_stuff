#!/usr/bin/env python

import time

f = open('fileopen_sleep.dat', 'w')
f.write('test\n')
f.close()

print 'sleep(5) start'

time.sleep(5)
print 'terminated'
