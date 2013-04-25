#!/usr/bin/env python

fpath = '/etc/dhcp3/dhcpd.conf'
hosts = {}

for block in open(fpath,'r').read().split('host '):
	words = block.split()
	try:
		mac = words[words.index('ethernet') + 1].rstrip(';')
		hostname = words[words.index('host-name') + 1][1:-2]
		hosts[hostname] = mac
	except ValueError:
		pass

for key, val in hosts.items():
	print(key, val)
