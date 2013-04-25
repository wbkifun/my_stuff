#!/usr/bin/env python


def conversion_byte_unit(b1):
	sizes = [1024**4, 1024**3, 1024**2, 1024]
	units = ['TB', 'GB', 'MB', 'KB']
	widths = [100, 10]

	for size, unit in zip(sizes, units):
		if b1 >= size:
			b1 = float(b1) / size
			ut = unit
			break

	if b1 >= 100: return ' %d %s' % (int(b1), ut)
	elif b1 >= 10: return '%2.1f %s' % (b1, ut)
	else: return '%1.2f %s' % (b1, ut)


mems = [7687, 83637, 736287, 6567898, 6543212, 34567898, 765432345, 76543356789, 987654323456, 34761973643864]
for mem in mems:
	print conversion_byte_unit(mem), mem
