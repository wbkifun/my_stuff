#!/bin/bash

if [ -f 'fib[1-4].so' ]; then 
	rm fib[1-1].so
fi

# f77
f2py -m fib1 -c fib1.f

f2py -m fib2 -h fib1.pyf fib1.f
f2py -c fib2.pyf fib1.f
f2py -c fib3.pyf fib1.f

f2py -m fib4 -c fib4.f

# f95
#f2py -m fib_f -c --fcompiler=gnu95 fibonacci.f95
