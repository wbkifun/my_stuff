from __future__ import division
import sympy
import numpy

print('-'*80)
print('sympy with character variable')
print('-'*80)
M = sympy.matrices.Matrix([[1,1,0],[1,'-a',0],[0,1,1]])
Minv = M.inv()
print M
print Minv


print('-'*80)
print('sympy with numeric variable')
print('-'*80)
M = sympy.matrices.Matrix([[1,1,0],[1,-1,0],[0,1,1]])
Minv = M.inv()
print M
print Minv


print('-'*80)
print('numpy with numeric variable')
print('-'*80)
M = numpy.matrix([[1,1,0],[1,-1,0],[0,1,1]])
Minv = numpy.linalg.inv(M)
print M
print Minv


print('-'*80)
print('numerical result: M x Minv')
print('-'*80)
print numpy.dot(M,Minv)
