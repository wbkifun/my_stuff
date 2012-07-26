#!/usr/bin/env python

from util_functions import *

def compile_c_files(dir, file):
	current_dir = os.getcwd()
	os.chdir(dir)
	o_file = file.replace('.c', '.o')
	so_file = file.replace('.c', '.so')

	print 'compile,link %s' %(dir+'/'+file)
	cmd = 'gcc -O3 -fpic -g -I. -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include -shared %s -o %s' %(file, so_file)
	os.system(cmd)

	os.chdir(current_dir)
	
	
if __name__ == '__main__':
	import sys
	try:
		mydir = sys.argv[1]
	except IndexError:
		mydir = '.'
		
	suffix_list = ['.c']
	function = compile_c_files
	
	matched_file_count = \
			find_recursively_matched_files( \
			mydir, suffix_list, function, False)
			
	if matched_file_count == 0:
		print 'No matched files.'
	else:
		opt = raw_input('These files are compile and link!\nDo you really? [Y/n]: ')
		if opt == '' or opt == 'y' or opt == 'Y':
			find_recursively_matched_files( \
					mydir, suffix_list, function, True)
			print 'Done!'
		else:
			print 'Canceled!'
