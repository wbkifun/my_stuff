#!/usr/bin/env python

#------------------------------------------------------------------------------
# File Name : node_updown.py
#
# Author : Kim Ki-hwan (wbkifun@korea.ac.kr)
# 
# Written date :	2008. 8. 12
# Modify date :		2008. 10. 30	add the node_reboot()		
#                   2011. 10. 19    modify path_dhcpd_conf, replace ssh to rsh
#
# Copyright : GNU GPLv2
#
# <File Description>
# Boot up/down the cluster nodes
#------------------------------------------------------------------------------


import os
import sys
import time


# set the parameters
base_ip = '192.168.30.'
ethernet_interface = 'eth0'
interval_time = 20	# second
path_dhcpd_conf = '/etc/dhcp/dhcpd.conf'


def print_usage():
	print 'Usage: ./node_updown.py boot_opt node_ip'
	print '\tboot_opt: boot/halt/reboot'
	print '\tnode_ip : 101\n\t\t  101,103,107\n\t\t  101-110\n\t\t  101,102,120-125\n\t\t  101-107,120-125\n'


def node_bootup(ip_list, mac_list):
	for i, ip in enumerate(ip_list):
		print 'Sending the magic packet to node %s.' % ip
		print 'wait %d seconds...' % interval_time
		for j in xrange(10):
			cmd = 'etherwake -i %s %s' % (ethernet_interface, mac_list[i])
			os.system(cmd)
		time.sleep(interval_time)


def node_halt(ip_list):
	for ip in ip_list:
		print 'Sending the halt command to node %s.' % ip
		cmd = 'rsh %s /sbin/halt' % (base_ip + ip)
		print cmd
		os.system(cmd)


def node_reboot(ip_list):
	for ip in ip_list:
		print 'Sending the reboot command to node %s.' % ip
		cmd = 'rsh %s /sbin/reboot' % (base_ip + ip)
		print cmd
		os.system(cmd)
		time.sleep(interval_time)


if __name__ == '__main__':
	#--------------------------------------------
	try:
		opt = sys.argv[1]
		node_ip = sys.argv[2]
	except IndexError:
		print_usage()
		sys.exit()

	print 'base_ip: ', base_ip
	print 'operation: ', opt


	#--------------------------------------------
	# make the ip_list
	ip_list = []
	for ip_str in node_ip.split(','):
		if '-' in ip_str:
			start_ip, end_ip = ip_str.split('-')
			start_ip = int( start_ip )
			end_ip = int( end_ip )
			for ip in xrange(start_ip, end_ip+1):
				ip_list.append( str(ip) )
		else:
			ip_list.append( ip_str )

	print 'IP_list: ', ip_list


	#--------------------------------------------
	# make the mac_list from the dhcpd.conf file
	mac_list = []
	read_file = file(path_dhcpd_conf)
	line_list = read_file.readlines()
	#print line_list

	for i, ip in enumerate( ip_list ):
		ip_address = base_ip + ip
		#print ip_address

		for j, line in enumerate( line_list ):
			if ip_address in line:
				mac = line_list[j-1][20:-3]
				mac_list.append( mac )

		if len(mac_list) != i+1:
			print 'Error: Not found the MAC address matched %s' % ip_address
			sys.exit()
	
	print 'MAC_list: ', mac_list


	#--------------------------------------------
	# execute the function
	if opt == 'boot':
		node_bootup(ip_list, mac_list)
	elif opt == 'halt':
		node_halt(ip_list)
	elif opt == 'reboot':
		node_reboot(ip_list)
	else:
		print_usage()
		sys.exit()
