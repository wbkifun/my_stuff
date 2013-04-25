#!/usr/bin/env python

def f(a, b):
	print a, b


def g(f, *arg):
	print arg
	#print *arg
	f(*arg)


def h(f, **kwd):
	print kwd


f(1,2)
g(f,1,2)

h(f,a=1)
