#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy
import sys
sys.path.append('./')
import ext_arr1d
import scipy


if __name__ == '__main__':
	N	=	3**3
	A	=	numpy.ones( (3,3,3) , dtype='f' )
	B	=	numpy.zeros( (3,3,3), 'f' )

	print ext_arr1d.__doc__
	ext_arr1d.arr1d(N,A,B)
	print B
