import numpy as np
from numpy import pi, sin, cos, tan, sqrt
from numpy.random import rand, randint
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_

import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
sys.path.append(current_dpath)




def test_ij2ab():
    '''
    ij2ab(): center of panel, at panel border
    '''
    from cs_ll import ij2ab

    alpha, beta = ij2ab(ne=16, ngq=4, ei=1, ej=1, gi=1, gj=1)
    a_equal([alpha,beta], [-pi/4,-pi/4])

    alpha, beta = ij2ab(ne=16, ngq=4, ei=8, ej=8, gi=4, gj=4)
    a_equal([alpha,beta], [0,0])

    #------------------------------------------------
    # MVP accuracy test
    #------------------------------------------------
    ne, ngq = 120, 4
    panel = 2
    gi1, gj1, ei1, ej1 = 4, 4, 84, 79
    gi2, gj2, ei2, ej2 = 1, 4, 85, 79
    gi3, gj3, ei3, ej3 = 1, 1, 85, 80
    gi4, gj4, ei4, ej4 = 4, 1, 84, 80

    a1, b1 = ij2ab(ne, ngq, ei1, ej1, gi1, gj1)
    a2, b2 = ij2ab(ne, ngq, ei2, ej2, gi2, gj2)
    a3, b3 = ij2ab(ne, ngq, ei3, ej3, gi3, gj3)
    a4, b4 = ij2ab(ne, ngq, ei4, ej4, gi4, gj4)

    #print('')
    #print('{:.15f}, {:.15f}'.format(a1, b1))
    #print('{:.15f}, {:.15f}'.format(a2, b2))

    aa_equal([a1,b1], [a2,b2], 15)
    aa_equal([a2,b2], [a3,b3], 15)
    aa_equal([a3,b3], [a4,b4], 15)
    aa_equal([a1,b1], [a4,b4], 15)




def test_abp2latlon():
    '''
    abp2latlon(): MVP low accuracy
    '''
    from cs_ll import ij2ab, abp2latlon

    ne, ngq = 120, 4
    panel = 2
    gi1, gj1, ei1, ej1 = 4, 4, 84, 79
    gi2, gj2, ei2, ej2 = 1, 4, 85, 79
    gi3, gj3, ei3, ej3 = 1, 1, 85, 80
    gi4, gj4, ei4, ej4 = 4, 1, 84, 80

    a1, b1 = ij2ab(ne, ngq, ei1, ej1, gi1, gj1)
    a2, b2 = ij2ab(ne, ngq, ei2, ej2, gi2, gj2)
    a3, b3 = ij2ab(ne, ngq, ei3, ej3, gi3, gj3)
    a4, b4 = ij2ab(ne, ngq, ei4, ej4, gi4, gj4)


    lat1, lon1 = abp2latlon(a1,b1,panel)
    lat2, lon2 = abp2latlon(a2,b2,panel)
    lat3, lon3 = abp2latlon(a3,b3,panel)
    lat4, lon4 = abp2latlon(a4,b4,panel)

    '''
    print('')
    print('{:.18f}, {:.18f}'.format(a1, b1))
    print('{:.18f}, {:.18f}'.format(a2, b2))
    #print('{:.18f}, {:.18f}'.format(a3, b3))
    #print('{:.18f}, {:.18f}'.format(a4, b4))
    print('')
    print('{:.18f}, {:.18f}'.format(lat1, lon1))
    print('{:.18f}, {:.18f}'.format(lat2, lon2))
    #print('{:.18f}, {:.18f}'.format(lat3, lon3))
    #print('{:.18f}, {:.18f}'.format(lat4, lon4))
    '''

    aa_equal([lat1,lon1], [lat2,lon2], 15)
    aa_equal([lat2,lon2], [lat3,lon3], 15)
    aa_equal([lat3,lon3], [lat4,lon4], 15)
    aa_equal([lat1,lon1], [lat4,lon4], 15)




def test_abp2latlon_2():
    '''
    abp2latlon(): check (ne=120, ei=84, ej=79, panel=2)
    '''
    from cs_ll import ij2ab, abp2latlon, latlon2abp

    ne, ngq = 120, 4
    panel = 2
    gi1, gj1, ei1, ej1 = 4, 4, 84, 79
    gi2, gj2, ei2, ej2 = 1, 4, 85, 79
    gi3, gj3, ei3, ej3 = 1, 1, 85, 80
    gi4, gj4, ei4, ej4 = 4, 1, 84, 80

    a1, b1 = ij2ab(ne, ngq, ei1, ej1, gi1, gj1)
    a2, b2 = ij2ab(ne, ngq, ei2, ej2, gi2, gj2)
    a3, b3 = ij2ab(ne, ngq, ei3, ej3, gi3, gj3)
    a4, b4 = ij2ab(ne, ngq, ei4, ej4, gi4, gj4)

    lat1, lon1 = abp2latlon(a1,b1,panel)
    lat2, lon2 = abp2latlon(a2,b2,panel)
    lat3, lon3 = abp2latlon(a3,b3,panel)
    lat4, lon4 = abp2latlon(a4,b4,panel)

    aa_equal([lat1,lon1], [lat2,lon2], 15)
    aa_equal([lat3,lon3], [lat2,lon2], 15)
    aa_equal([lat3,lon3], [lat4,lon4], 15)
    aa_equal([lat1,lon1], [lat4,lon4], 15)




def test_latlon2xyp():
    '''
    latlon2xyp(): center of panel, at panel border
    '''
    from cs_ll import latlon2xyp, xyp2latlon

    R = 1
    a = R/sqrt(3)
    rlat, rlon = 0, 0

    #------------------------------------------------
    # center of panel
    #------------------------------------------------
    xyp_dict = latlon2xyp(0, 0, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [1])
    aa_equal(list(xyp_dict.values()), [(0,0)], 15)

    xyp_dict = latlon2xyp(0, pi/2, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [2])
    aa_equal(list(xyp_dict.values()), [(0,0)], 15)

    xyp_dict = latlon2xyp(0, pi, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [3])
    aa_equal(list(xyp_dict.values()), [(0,0)], 15)

    xyp_dict = latlon2xyp(0, 3*pi/2, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [4])
    aa_equal(list(xyp_dict.values()), [(0,0)], 15)

    xyp_dict = latlon2xyp(-pi/2, 0, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [5])
    aa_equal(list(xyp_dict.values()), [(0,0)], 15)

    xyp_dict = latlon2xyp(pi/2, 0, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [6])
    aa_equal(list(xyp_dict.values()), [(0,0)], 15)


    #------------------------------------------------
    # at the panel border
    #------------------------------------------------
    alpha = pi/4
    r_cos, r_sin = R*cos(alpha), R*sin(alpha)
    at = a*tan(alpha)

    xyp_dict = latlon2xyp(0, pi/4, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [1,2])
    aa_equal(list(xyp_dict.values()), [(at,0), (-at,0)], 15)

    xyp_dict = latlon2xyp(0, 3*pi/4, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [2,3])
    aa_equal(list(xyp_dict.values()), [(at,0), (-at,0)], 15)

    xyp_dict = latlon2xyp(0, 5*pi/4, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [3,4])
    aa_equal(list(xyp_dict.values()), [(at,0), (-at,0)], 15)

    xyp_dict = latlon2xyp(0, 7*pi/4, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [1,4])
    aa_equal(list(xyp_dict.values()), [(-at,0), (at,0)], 15)

    xyp_dict = latlon2xyp(-pi/4, pi/2, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [2,5])
    aa_equal(list(xyp_dict.values()), [(0,-at), (at,0)], 15)

    xyp_dict = latlon2xyp(pi/4, pi/2, rlat, rlon)
    a_equal(list(xyp_dict.keys()), [2,6])
    aa_equal(list(xyp_dict.values()), [(0,at), (at,0)], 15)




def test_latlon2xyp_xyp2latlon():
    '''
    latlon2xyp() -> xyp2latlon() : check consistency, repeat 1000 times
    '''
    from cs_ll import latlon2xyp, xyp2latlon


    N = 1000
    R = 1
    a = R/sqrt(3)

    for i in range(N):
        lat = pi*rand() - pi/2
        lon = 2*pi*rand()

        xyp_dict = latlon2xyp(lat, lon)

        for panel, (x,y) in xyp_dict.items():
            lat2, lon2 = xyp2latlon(x, y, panel)
            aa_equal((lat,lon), (lat2,lon2), 12)




def print_low_accuracy():
    '''
    xyp -> latlon -> xyp: low accuracy
    '''
    from cs_ll import latlon2xyp, xyp2latlon
    from cart_ll import latlon2xyz, xyz2latlon
    from cart_cs import xyp2xyz, xyz2xyp


    #------------------------------------------------
    # at the panel border, low_accuracy
    #------------------------------------------------
    xyp1 = (-0.57735026918962606, -0.53747283769483301, 1)
    xyp2 = (-0.57735026918962495, -0.53747283769483301, 1)

    latlon1 = xyp2latlon(*xyp1)
    latlon2 = xyp2latlon(*xyp2)
    xyp1_dict = latlon2xyp( *xyp2latlon(*xyp1) )
    xyp2_dict = latlon2xyp( *xyp2latlon(*xyp2) )

    print('')
    print(repr(latlon1))
    print(repr(latlon2))
    print(repr(xyp1_dict[1]))
    print(repr(xyp2_dict[1]))

    #a_equal(xyp1_dict.keys(), [1,4])
    #a_equal(xyp2_dict.keys(), [1,4])


    xyz1 = xyp2xyz(*xyp1)
    xyz2 = xyp2xyz(*xyp2)
    xyp1_list = xyz2xyp(*xyz1)
    xyp2_list = xyz2xyp(*xyz2)
    print('')
    print(repr(xyz1))
    print(repr(xyz2))
    print(repr(xyp1_dict[1]), xyp1_dict.keys())
    print(repr(xyp2_dict[1]), xyp1_dict.keys())


    a = 1/np.sqrt(3)
    at1, at2 = a*np.tan(-np.pi/4), a*np.tan(np.pi/4) 
    print('')
    print(repr(at1), repr(at2))
