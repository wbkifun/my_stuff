#!/usr/bin/env python

import subprocess as sp

remote_hosts = ['g101']
path = '/home/kifang/test.py'

for host in remote_hosts:
	cmd = 'rsh %s python %s' % (host, path)
	proc = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE)
	print 'aaa'
	stdout, stderr = proc.communicate()
	print 'bbb'
	print('%s' % host)
	print('stdout: %s' % stdout)
	print('stderr: %s' % stderr)
