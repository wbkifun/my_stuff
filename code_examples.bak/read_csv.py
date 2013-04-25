#!/usr/bin/env python

import csv
import numpy as np

def float_array_from_csv(filename, skip_header=True):
	f = open(filename)
	try:
		reader = csv.reader(f)
		output_list = []
		if skip_header:
			reader.next()
		for row in reader:
			print row
			output_list.append(map(float, row))
	finally:
		f.close()

	return np.array(output_list)

a = float_array_from_csv('./read_csv_data-gold.csv', True)
print a
