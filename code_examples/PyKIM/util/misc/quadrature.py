#------------------------------------------------------------------------------
# filename  : quadrature.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.8.23  add gq_integrate_2d()
#             2013.9.3   use the default parameter
#             2014.2.20  modify for high order legendre
#             2014.3.24  change to class and generate preprocess file
#
# description:
#   subroutines about the Gauss-Quadrature
#
# subroutines:
#   legendre()          return a value of Legendre function
#   deriv_legendre()    1st-derivative Legendre function
#   recursive_L_dL()    generate the Legendre and 1st derivatives recursively
#                       not use because of low accuracy
#   gausslobatto()      generate the Gauss-Legendre-Lobatto points and weights
#   quad_norm()
#   gq_integrate()      definite integral of a given one-dimensional function
#   gq_integrate_2d()   definite integral of a given two-dimensional function
#
# class:
#   GQIntegrate
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import pi, cos, sqrt
from math import fsum
import pickle
import os
import sys

from compare_float import feq


DTYPE = np.float128
PI = DTYPE('3.1415926535897932384626433832795028')
TOL = 4*np.finfo(DTYPE).eps
MAX_P_ORDER = 8





def legendre(p_order, x):
    L = [None for i in xrange(MAX_P_ORDER+1)]
    L[0] = lambda x: 1
    L[1] = lambda x: x
    L[2] = lambda x: 1/2*(3*x**2 - 1)
    L[3] = lambda x: 1/2*(5*x**3 - 3*x)
    L[4] = lambda x: 1/8*(35*x**4 - 30*x**2 + 3)
    L[5] = lambda x: 1/8*(63*x**5 - 70*x**3 + 15*x)
    L[6] = lambda x: 1/16*(231*x**6 - 315*x**4 + 105*x**2 - 5)
    L[7] = lambda x: 1/16*(429*x**7 - 693*x**5 + 315*x**3 - 35*x)
    L[8] = lambda x: 1/128*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35)

    if p_order > MAX_P_ORDER:
        #raise 'Must be p_order <= %d' % MAX_P_ORDER
        L, dL = recursive_L_dL(p_order, x)
        return L
    else:
        return L[p_order](x)




def deriv_legendre(p_order, x):
    dL = [None for i in xrange(MAX_P_ORDER+1)]
    dL[0] = lambda x: 0
    dL[1] = lambda x: 1
    dL[2] = lambda x: 1/2*(6*x)
    dL[3] = lambda x: 1/2*(15*x**2 - 3)
    dL[4] = lambda x: 1/8*(140*x**3 - 60*x)
    dL[5] = lambda x: 1/8*(315*x**4 - 210*x**2 + 15)
    dL[6] = lambda x: 1/16*(1386*x**5 - 1260*x**3 + 210*x)
    dL[7] = lambda x: 1/16*(3003*x**6 - 3465*x**4 + 945*x**2 - 35)
    dL[8] = lambda x: 1/128*(51480*x**7 - 72072*x**5 + 27720*x**3 - 2520*x)

    if p_order > MAX_P_ORDER:
        #raise 'Must be p_order <= %d' % MAX_P_ORDER
        L, dL = recursive_L_dL(p_order, x)
        return dL
    else:
        return dL[p_order](x)




def recursive_L_dL(p_order, x):
    '''
    Recursive formula for Legendre polynomials and derivatives
    Note that the precision is cutted.
    '''
    L = [None for i in xrange(3)]
    dL = [None for i in xrange(3)]

    if p_order == 0:
        L[0] = 1
        dL[0] = 0

    elif p_order == 1:
        L[0] = x
        dL[0] = 1

    else:
        L[-2], L[-1] = 1, x
        dL[-2], dL[-1] = 0, 1

        for k in xrange(2,p_order+1):
            L[0] = ((2*k-1)/k)*x*L[-1] - ((k-1)/k)*L[-2]
            dL[0] = dL[-2] + (2*k-1)*L[-1]
            L[-2], L[-1] = L[-1], L[0]
            dL[-2], dL[-1] = dL[-1], dL[0]

    return L[0], dL[0]




def gausslobatto(p_order, max_iter=10000):
    '''
    Find the Gauss-Lobatto-Legendre collocation points xgl(i)
    and the corresponding weights.
    '''

    fname = 'gausslobatto_x_w_list_200.pkl'
    if os.path.isfile(fname):
        f = open(fname, 'rb')
        x_w_list = pickle.load(f)

        return x_w_list[p_order]


    N = p_order
    x = np.zeros(N+1, dtype=DTYPE)
    w = np.zeros(N+1, dtype=DTYPE)

    if N == 1:
        x[0], x[1] = -1, 1
        w[0], w[1] = 1, 1

    else:
        x[0], x[N] = -1, 1
        w[0], w[N] = 2/(N*(N+1)), 2/(N*(N+1))

        for j in xrange(1, (N+1)//2+1):
            x[j] = -cos( (j+1/4)*PI/N - 3/(8*N*PI*(j+1/4)) )

            for k in xrange(max_iter):
                q = legendre(N+1, x[j]) - legendre(N-1, x[j])
                dq = deriv_legendre(N+1, x[j]) - deriv_legendre(N-1, x[j])
                delta = -q/dq
                x[j] += delta

                if abs(delta) <= TOL*abs(x[j]): 
                    break

            L = legendre(N, x[j])
            x[N-j] = -x[j]
            w[j] = 2/(N*(N+1)*L**2)
            w[N-j] = w[j]

    if np.mod(N,2) == 0:
        L = legendre(N,0)
        x[N//2] = 0
        w[N//2] = 2/(N*(N+1)*L**2)

    return x, w




def gausslegendre(p_order, max_iter=10000):
    '''
    Find the Gauss-Legendre collocation points xgl(i)
    and the corresponding weights.
    '''

    fname = 'gausslegendre_x_w_list_200.pkl'
    if os.path.isfile(fname):
        print 'read gausslegendre file'
        f = open(fname, 'rb')
        x_w_list = pickle.load(f)

        return x_w_list[p_order]
    else:
        print 'file not found'



    N = p_order
    x = np.zeros(N+1, dtype=DTYPE)
    w = np.zeros(N+1, dtype=DTYPE)

    if N == 0:
        x[0] = 0
        w[0] = 2

    elif N == 1:
        x[0], x[1] = -sqrt(1/3), sqrt(1/3)
        w[0], w[1] = 1, 1

    else:
        for j in xrange(0, (N+1)//2+1):
            x[j] = -cos( PI*(2*j+1)/(2*N+2) )

            for k in xrange(max_iter):
                L = legendre(N+1, x[j])
                dL = deriv_legendre(N+1, x[j])
                delta = -L/dL
                x[j] += delta

                if abs(delta) <= TOL*abs(x[j]): 
                    break

            L = legendre(N+1, x[j])
            dL = deriv_legendre(N+1, x[j])
            x[N-j] = -x[j]
            w[j] = 2/( (1-x[j]*x[j]) * dL*dL )
            w[N-j] = w[j]

    if np.mod(N,2) == 0:
        L = legendre(N+1,0)
        dL = deriv_legendre(N+1,0)
        x[N//2] = 0
        w[N//2] = 2/(dL*dL)

    return x, w




def quad_norm(n, gll_pts, gll_wts):
    '''
    Compute normalization constants
    for k=1,n order Legendre polynomials

    e.g. gamma[k] in Canuto, page 58.
    '''

    gamma = np.zeros(n, dtype=DTYPE)

    for i in xrange(n):
        for j in xrange(n):
            gamma[j] += gll_wts[i] * legendre(j, gll_pts[i])**2

    return gamma




#==============================================================================




class GQIntegrate(object):
    def __init__(self):
        basedir = os.path.dirname(__file__) + '/'

        fname = 'gausslobatto_x_w_list_200.pkl'
        fpath = basedir + fname
        if os.path.isfile(fpath):
            f = open(fpath, 'rb')
            self.gausslobatto_x_w_list = pickle.load(f)
        else:
            yn = raw_input('The file %s is not found. Generate? (Y/n) '%fname)
            if yn in ['n','N']:
                sys.exit()
            else:
                x_w_list = [None]
                for p_order in xrange(1,201):
                    x, w = gausslobatto(p_order)
                    x_w_list.append( (x, w) )

                f = open(fpath, 'wb')
                pickle.dump(x_w_list, f)
                f.close()


        fname = 'gausslegendre_x_w_list_200.pkl'
        fpath = basedir + fname
        if os.path.isfile(fpath):
            f = open(fpath, 'rb')
            self.gausslegendre_x_w_list = pickle.load(f)
        else:
            yn = raw_input('The file %s is not found. Generate? (Y/n) '%fname)
            if yn in ['n','N']:
                sys.exit()
            else:
                x_w_list = [None]
                for p_order in xrange(1,201):
                    x, w = gausslegendre(p_order)
                    x_w_list.append( (x, w) )

                f = open(fpath, 'wb')
                pickle.dump(x_w_list, f)
                f.close()



    def gq_integrate(self, x1, x2, func, p_order=7, qtype='lobatto'):
        if qtype == 'lobatto':
            gll_pts, gll_wts = self.gausslobatto_x_w_list[p_order]
        elif qtype == 'legendre':
            gll_pts, gll_wts = self.gausslegendre_x_w_list[p_order]
        else:
            raise ValueError, 'invalid qtype %s'%qtype

        M, N = 0.5*(x2+x1), 0.5*(x2-x1)

        sum_arr = np.zeros(p_order+1)
        for i, (x, w) in enumerate( zip(gll_pts, gll_wts) ):
            sum_arr[i] = w*func(M+N*x)

        return N*fsum(sum_arr)




    def gq_integrate_2d(self, x1, x2, y1, y2, func, p_order=7, qtype='lobatto'):
        if qtype == 'lobatto':
            gll_pts, gll_wts = self.gausslobatto_x_w_list[p_order]
        elif qtype == 'legendre':
            gll_pts, gll_wts = self.gausslegendre_x_w_list[p_order]
        else:
            raise ValueError, 'invalid qtype %s'%qtype

        M, N = 0.5*(x2+x1), 0.5*(x2-x1)
        P, Q = 0.5*(y2+y1), 0.5*(y2-y1)

        sum_arr = np.zeros(((p_order+1),(p_order+1)))
        for i, (x, wx) in enumerate( zip(gll_pts, gll_wts) ):
            for j, (y, wy) in enumerate( zip(gll_pts, gll_wts) ):
                sum_arr[i,j] = wx*wy*func(M+N*x,P+Q*y)

        return N*Q*fsum(sum_arr.flatten())




    def gq_integrate_iterate(self, x1, x2, func, qtype='lobatto'):
        gq_integrate = self.gq_integrate

        int1 = gq_integrate(x1, x2, func, 5, qtype)
        int2 = gq_integrate(x1, x2, func, 7, qtype)

        for p_order in range(10,101,5):
            int3 = gq_integrate(x1, x2, func, p_order, qtype)

            if feq(int1,int3) and feq(int2,int3):
                break
            else:
                int1, int2 = int2, int3

        return int3, p_order




    def gq_integrate_2d_iterate(self, x1, x2, y1, y2, func, qtype='lobatto'):
        gq_integrate = self.gq_integrate_2d

        int1 = gq_integrate_2d(x1, x2, y1, y2, func, 5, qtype)
        int2 = gq_integrate_2d(x1, x2, y1, y2, func, 7, qtype)

        for p_order in range(10,101,5):
            int3 = gq_integrate_2d(x1, x2, y1, y2, func, p_order, qtype)

            if feq(int1,int3) and feq(int2,int3):
                break
            else:
                int1, int2 = int2, int3

        return int3, p_order
