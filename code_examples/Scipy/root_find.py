from __future__ import division
import numpy
import scipy 
from scipy.optimize import fsolve



#------------------------------------------------------------------------------
# Linear equations
#
# x+3y+5z=10
# 2x+5y+z=8
# 2x+3y+8z=3
#------------------------------------------------------------------------------
print '-'*47
print 'Linear equations'
print '-'*47
A = numpy.mat([[1,3,5],[2,5,1],[2,3,8]])
b = numpy.mat([[10],[8],[3]])
det_A = numpy.linalg.det(A)
sol = numpy.linalg.solve(A,b)
print 'A', A
print 'b', b
print 'det(A)', det_A
print sol
print A*sol-b

a = numpy.array([[1,3,5],[2,5,1],[2,3,8]])
b = numpy.array([10,8,3])

x = numpy.linalg.solve(a,b)

print x



#------------------------------------------------------------------------------
# Find the roots of the polynomial
#
# x^3 - 3x^2 + 2x - 1 = 0
#------------------------------------------------------------------------------
print '\n','-'*47
print 'Find the roots of the polynomial'
print '-'*47
print 'x^3 - 3x^2 + 2x - 1 = 0'
poly = [1,-3,2,-1]
sol = numpy.roots(poly)
print sol
func = lambda x:x**3 - 3*x**2 + 2*x - 1
for s in sol: print func(s)




#------------------------------------------------------------------------------
# Find a root of the non-linear equation
#
# x + 2cos(x) = 0
#------------------------------------------------------------------------------
print '\n','-'*47
print 'Find a root of the non-linear equation'
print '-'*47
print 'x + 2cos(x) = 0'

from numpy import cos
func = lambda x: x+2*cos(x)

print '\nbisection method starting on the interval [-2,2]'
sol = scipy.optimize.bisect(func, -2, 2)
print sol, func(sol)

print '\nNewton-Raphson method with starting point at x0=2'
sol = scipy.optimize.newton(func, 2)
print sol, func(sol)

print '\nfsolve at starting point at x0=0.3'
sol = scipy.optimize.fsolve(func, 0.3)
print sol, func(sol)

print '\nbrentq method starting point on the interval [-2,2]'
sol = scipy.optimize.brentq(func, -2, 2)
print sol, func(sol)

print '\ncheck to include the end point with brentq method'
sol = scipy.optimize.brentq(func, sol, 2)
print sol, func(sol)




#------------------------------------------------------------------------------
# Find a root of the 2nd-degree polynomial
#
# (x+0.5)(x-0.5) = 0
#------------------------------------------------------------------------------
print '\n','-'*47
print 'Find a root of the 2nd-degree polynomial'
print '-'*47
print '(x+0.5)(x-0.5) = 0'

func = lambda x: (x+0.5)*(x-0.5)

print '\ncheck to include the end point with bentq method'
sol = scipy.optimize.brentq(func, -1, 0)
print sol, func(sol)

sol = scipy.optimize.brentq(func, sol, 1)
print sol, func(sol)

sol = scipy.optimize.brentq(func, sol+0.0001, 1)
print sol, func(sol)




#------------------------------------------------------------------------------
# Find a root of the system of non-linear equations
#
# x cos(y) = 4
# xy-y = 5
#------------------------------------------------------------------------------
print '\n','-'*47
print 'Find a root of the system of non-linear equations'
print '-'*47
print 'x cos(y) = 4'
print 'xy-y = 5'

func = lambda x: (x[0]*cos(x[1])-4, x[0]*x[1]-x[1]-5)
sol = scipy.optimize.fsolve(func, [1,1], xtol=1e-15)
print sol, func(sol)
