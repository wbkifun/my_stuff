#------------------------------------------------------------------------------
# filename  : test_cube_vtk.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.18     start
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np

from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_, with_setup

from cube_vtk import CubeVTK2D




def test_cubevtk2d():
    '''
    CubeVTK2D: ne=3, ngq=4
    '''
    ne, ngq = 3, 4
    cs_vtk = CubeVTK2D(ne, ngq)

    gq_indices = cs_vtk.cs_ncf.variables['gq_indices'][:]
    uids = cs_vtk.cs_ncf.variables['uids'][:]
    ij2uid = dict()
    for seq, ij in enumerate(gq_indices):
        ij2uid[tuple(ij)] = uids[seq]

    a_equal(cs_vtk.links[0], [9,0,1,5,4])
    a_equal(cs_vtk.links[2], [9,2,3,7,6])

    link = [(1,1,1,4,1), (1,2,1,2,1), (1,2,1,2,2), (1,1,1,4,2)]
    a_equal(cs_vtk.links[9], [9]+[ij2uid[ij] for ij in link])

    link = [(1,1,1,4,2), (1,2,1,2,2), (1,2,1,2,3), (1,1,1,4,3)]
    a_equal(cs_vtk.links[12], [9]+[ij2uid[ij] for ij in link])

    link = [(1,1,1,4,4), (1,2,1,2,4), (1,2,2,2,2), (1,1,2,4,2)]
    a_equal(cs_vtk.links[36], [9]+[ij2uid[ij] for ij in link])

    link = [(1,3,1,4,1), (2,1,1,2,1), (2,1,1,2,2), (1,3,1,4,2)]
    a_equal(cs_vtk.links[81], [9]+[ij2uid[ij] for ij in link])

    link = [(4,3,3,4,1), (1,1,3,2,1), (1,1,3,2,2), (1,1,3,1,2)]
    a_equal(cs_vtk.links[54], [9]+[ij2uid[ij] for ij in link])
