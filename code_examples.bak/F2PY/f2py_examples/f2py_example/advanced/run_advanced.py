import numpy as N
import pylab as P

import simple_ice_model_advanced

# import os
# import sys
# sys.path.append('..')
# from py_fortran_tools import *

def run():
    grid = 51
    dt = 0.1
    t_final = 1000

    (thick, bed, xx, t_out) = simple_ice_model_advanced.simple_ice_model(b_cond=[0,0])

    P.figure()
    P.plot(xx/1e3, bed+thick[:,-1], 'b')
    P.hold(True)
    P.plot(xx/1e3, bed, 'r')
    P.xlabel('x (km)')
    P.ylabel('b+H (m)')

if __name__=='__main__':
    run()
    P.show()
