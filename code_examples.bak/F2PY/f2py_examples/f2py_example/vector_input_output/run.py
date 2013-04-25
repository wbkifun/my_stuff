import numpy as N
import vector_in_out as vio

print 'DOC MODULE'
print vio.__doc__
print 'DOC FUNCTION'
print vio.vec_in_out.__doc__

matrix = N.random.rand(4,5)
vec = N.random.rand(4)

print '\nIN MATRIX'
print matrix
print '\nIN-OUT VECTOR'
print vec

out = vio.vec_in_out(matrix, vec)

print '\nexecuting Fortran module\n'

print '\nIN MATRIX'
print matrix
print '\nIN-OUT VECTOR'
print vec
print '\nOUT VECTOR'
print out
