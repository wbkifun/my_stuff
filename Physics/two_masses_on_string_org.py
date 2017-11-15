#------------------------------------------------------------------------------
# filename  : two_masses_on_string.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.10.11    Start
#
#
# Description: 
#   Find angles and tensions of two masses on a string
#   Solve a physics problem using Newton-Rapson method and Matrix solver
#
# Reference
#   Computational Physics: Problem Solving with Python, 3rd Edition
#   Rubin H. Landau, Manuel J Páez, Cristian C. Bordeianu 
#------------------------------------------------------------------------------

import numpy as np
import numpy.linalg as la




n = 9
eps = 1e-3
deriv = np.zeros((n,n))
f = np.zeros(n)
x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1.])




def F(x, f):
    f[0] = 3*x[3] + 4*x[4] + 4*x[5] - 8.0
    f[1] = 3*x[0] + 4*x[1] - 4*x[2]
    f[2] = x[6]*x[0] - x[7]*x[1] - 10.0
    f[3] = x[6]*x[3] - x[7]*x[4]
    f[4] = x[7]*x[1] + x[8]*x[2] - 20.0
    f[5] = x[7]*x[4] - x[8]*x[5]
    f[6] = x[0]**2 + x[3]**2 - 1.0
    f[7] = x[1]**2 + x[4]**2 - 1.0
    f[8] = x[2]**2 + x[5]**2 - 1.0




def dFdx(x, deriv, n):
    h = 1e-4

    for j in range(n):
        tmp = x[j]
        x[j] += h/2
        F(x, f)
        for i in range(n): deriv[i,j] = f[i]
        x[j] = tmp

    for j in range(n):
        tmp = x[j]
        x[j] -= h/2
        F(x, f)
        for i in range(n): deriv[i,j] = (deriv[i,j] - f[i])/h
        x[j] = tmp




#
# Newton-Rapson method
#
for it in range(100):
    #print('{:3d}'.format(it), flush=True, end='')
    F(x, f)
    dFdx(x, deriv, n)
    b = np.array([[-f[i]] for i in range(n)])
    dx = la.solve(deriv, -f)
    x[:] += dx[:]
    errX = errF = errXi = 0.0
    for i in range(n):
        if x[i] != 0: errXi = abs(dx[i]/x[i])
        else:         errXi = abs(dx[i])
        if errXi > errX: errX = errXi
        if abs(f[i]) > errF: errF = abs(f[i])
    if (errX <= eps) and (errF <= eps): break

print('Number of iterations =', it+1)
print('Solution =\n',x)
