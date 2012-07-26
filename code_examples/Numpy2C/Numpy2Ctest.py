#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import numpy
import sys
sys.path.append('./')
import ext_matmul
import scipy

def matmul0(N,A,B,C):
	for i in xrange(N):
		for j in xrange(N):
			stmp = 0
			for k in xrange(N):
				stmp += A[i,k]*B[k,j]
				
				C[i,j] = stmp


if __name__ == '__main__':
	N	=	5
	A	=	numpy.zeros((N,N),dtype='f')
	B	=	numpy.zeros((N,N),'f')
	C	=	numpy.zeros((N,N),'f')
	D	=	numpy.zeros((N,N),'f')

	for i in xrange(N):
		for j in xrange(N):
			A[i,j]	=	i-2*j+1
			B[i,j]	=	i*j-i+1

	print A
	print B

	matmul0(N,A,B,C)
	print C

	print ext_matmul.__doc__
	ext_matmul.matmul(N,A,B,D)
	print D
