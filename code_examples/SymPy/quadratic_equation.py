from sympy import Symbol
from sympy import solve, roots
from sympy import simplify
from sympy import init_printing

init_printing()

t = Symbol('t')
x1 = Symbol('x1')
x2 = Symbol('x2')
x3 = Symbol('x3')
x4 = Symbol('x4')
y1 = Symbol('y1')
y2 = Symbol('y2')
y3 = Symbol('y3')
y4 = Symbol('y4')

f = ((1-t)*x3+t*x4-(x1+x2)/2)**2 + ((1-t)*y3+t*y4-(y1+y2)/2)**2 \
    - (x2-x1)**2 - (y2-y1)**2

s = solve(f, t)
print simplify(s[0])
#print s[1]
