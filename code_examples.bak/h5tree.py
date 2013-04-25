#!/usr/bin/env python

#------------------------------------------------------------
# File Name : h5tree.py
# Author : Ki-Hwan Kim (wbkifun@nate.com)
# Copyright : GNU GPL
#
# Version : 1.0
# Changelog : 
#	2011.4.11 - first release (v1.0)
#
# <File Description>
# Show the tree structure of a hdf5 file
#------------------------------------------------------------

import h5py as h5
import sys


def print_attrs(obj, prefix):
	for att_key, att_value in obj.attrs.items():
		print("%s attrs['%s'] = %s" %(prefix, att_key, att_value))


def print_group_dataset(group, depth=0):
	prefix = '  ' * depth + '|'
	group_list = []

	s = group.name
	if s == '/': print('/')
	else: print('%s %s/' %(prefix, s[s.rfind('/')+1:]))

	prefix = '  ' * (depth+1) + '|'
	print_attrs(group, prefix)
	for key, value in group.items():
		if type(group[key]) == h5.Dataset:
			print("%s ['%s'] : %s, %s" %(prefix, key, value.shape, value.dtype))
			print_attrs(value, prefix)

		elif type(group[key]) == h5.Group:
			group_list.append(group[key])
	
	for group in group_list:
		print_group_dataset(group, depth+1)


if __name__ == '__main__':
	try:
		fpath = sys.argv[1]
	except IndexError:
		print('Usage: h5tree.py some_hdf5_file.h5\n')
		sys.exit(0)
	f = h5.File(fpath, 'r')
	print_group_dataset(f)
	f.close()
