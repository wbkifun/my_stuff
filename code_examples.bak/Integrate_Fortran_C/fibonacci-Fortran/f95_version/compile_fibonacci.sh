#!/bin/bash

if [ -f 'fib[1-4].so' ]; then 
	rm fib[1-1].so
fi

# f95
f2py -m fib1 -c --fcompiler=gnu95 fib1.f95

f2py -m fib2 -h fib1.pyf fib1.f95
f2py -c --fcompiler=gnu95 fib2.pyf fib1.f95
f2py -c --fcompiler=gnu95 fib3.pyf fib1.f95

f2py -m fib4 -c --fcompiler=gnu95 fib4.f95
