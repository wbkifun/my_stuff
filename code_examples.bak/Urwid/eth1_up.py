#!/usr/bin/env python

ips = {'yhome1':'163.152.42.14',
	   'yhome2':'163.152.42.15',
	   'yhome3':'163.152.42.16'}
gateway = '163.152.42.1'

import subprocess as sp
stdout, stderr = sp.Popen(['hostname'], stdout=sp.PIPE, stderr=sp.PIPE).communicate()
cmd = 'ifconfig eth1 %s netmask 255.255.255.0 up' % ips[stdout]
sp.Popen(cmd.split())
cmd = 'route add -net default gw %s dev eth1' % gateway
sp.Popen(cmd.split())
