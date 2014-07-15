from sympy import Symbol
from sympy import solve, roots, solve_poly_system
from sympy import sqrt
from sympy import init_printing

init_printing()

'''
x = Symbol('x')
s = solve(x**3 + 2*x + 3, x)
print s
'''


x1 = Symbol('x1')
x2 = Symbol('x2')
y1 = Symbol('y1')
y2 = Symbol('y2')
x = Symbol('x')
y = Symbol('y')

f = (x2-x1)**2 + (y2-y1)**2 - ((x-x1)**2 + (y-y1)**2)*2
g = (x2-x1)**2 + (y2-y1)**2 - ((x-x2)**2 + (y-y2)**2)*2 
s = solve_poly_system([f,g],x,y)
print 'x:', s[0][0]
print 'x:', s[1][0]
print ''
print 'y:', s[0][1]
print 'y:', s[1][1]
