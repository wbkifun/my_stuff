from __future__ import division
import numpy
from numpy import sqrt
from math import fsum



def ste_1_2_inf(x0, x1):
    x0 = x0.flatten()
    x1 = x1.flatten()

    L1 = fsum( abs(x0-x1) )/fsum( abs(x0) )
    L2 = sqrt( fsum( (x0-x1)**2 ) )/sqrt( fsum( x0**2 ) )
    Linf = max( abs(x0-x1) )/ max( abs(x0) )

    return L1, L2, Linf
