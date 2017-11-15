from sympy import *

init_printing()
x, y = symbols('x y')
y = diff(tan(x), x)
print(y)
