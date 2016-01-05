#------------------------------------------------------------------------------
# filename  : voronoi.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.12.15    using scipy
#             2015.12.18    manually
#             2015.12.23    change the distance threshold to 1e-5 empirically
#
#
# Description: 
#   Generate Voronoi diagram locally
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from scipy.spatial import Voronoi

from util.convert_coord.cs_ll import latlon2xyp
from util.geometry.sphere import distance3, angle3
from util.geometry.circumcircle import circum_center_radius




def get_voronoi_scipy(lat0, lon0, neighbor_latlons):
    x0, y0 = latlon2xyp(lat0,lon0,lat0,lon0)[1]     # check (0,0)
    nbr_xys = [latlon2xyp(lat,lon,lat0,lon0)[1] for lat,lon in neighbor_latlons]


    #------------------------------------------
    # Generate a Voronoi diagram
    vor = Voronoi([(x0,y0)]+nbr_xys)


    #------------------------------------------
    # Remove non-neighbor points
    ridge_pts = [sorted(pair) for pair in vor.ridge_points]
    nbr_xys2 = [nbr_xys[i] for i in xrange(len(nbr_xys)) \
                           if [0,i+1] in ridge_pts]
    if len(nbr_xys2) < len(nbr_xys):
        vor = Voronoi([(x0,y0)]+nbr_xys2)

    #for nbr, xy in zip(nbrs, nbr_xys2):
    #    print nbr, self.gq_indices[nbr], xy


    #------------------------------------------
    # Sort the vertices
    ridge_pts = [sorted(pair) for pair in vor.ridge_points]
    vidxs = list()

    for inb in xrange(1,len(nbr_xys2)+1):
        seq = ridge_pts.index([0,inb])
        v1, v2 = vor.ridge_vertices[seq]

        if inb == 1:
            vidxs.append([v1,v2])

        elif inb == 2:
            vidx0 = np.intersect1d(vidxs[0], [v1,v2])[0]
            vidx1 = v1 if vidx0 == v2 else v2
            vidxs[0] = vidx0
            vidxs.append(vidx1)

        else:
            vidx = v1 if vidxs[-1] == v2 else v2
            vidxs.append(vidx)

    xy_vertices = vor.vertices[vidxs]


    return xy_vertices, vor




def get_voronoi_xyzs(xyz0, neighbor_xyzs):
    '''
    neighbor_xyzs should be sorted in anti-clockwise direction
    '''
    nbr_xyzs = neighbor_xyzs
    i_list = range(len(nbr_xyzs))
    i_min = sorted(i_list, key=lambda i:distance3(xyz0, nbr_xyzs[i]))[0]
    i3s = i_list[i_min:] + i_list[:i_min] + [i_min]


    # Select valid points to construct the Delaunay triangles
    valid_idxs = [i_min]
    for i1, i2, i3 in zip(i3s[:-2], i3s[1:-1], i3s[2:]):
        xyz1 = nbr_xyzs[i1]
        xyz2 = nbr_xyzs[i2]
        xyz3 = nbr_xyzs[i3]

        a02 = angle3(xyz1, xyz0, xyz3) + angle3(xyz1, xyz2, xyz3)
        a13 = angle3(xyz0, xyz1, xyz2) + angle3(xyz0, xyz3, xyz2)

        if a02 > a13: 
            valid_idxs.append(i2)


    # Circumcircles of Delaunay triangles
    i2s = valid_idxs + [i_min]
    c_xyzs = list()
    for i1, i2 in zip(i2s[:-1], i2s[1:]):
        xyz1 = nbr_xyzs[i1]
        xyz2 = nbr_xyzs[i2]

        c_xyz, radius = circum_center_radius(xyz0, xyz1, xyz2)
        c_xyzs.append(c_xyz)

    c2s = c_xyzs + [c_xyzs[0]]
    voronoi_xyzs = list()
    for c_xyz1, c_xyz2 in zip(c2s[:-1], c2s[1:]):
        if distance3(c_xyz1, c_xyz2) > 1e-5:    # empirical threshold
            voronoi_xyzs.append(c_xyz1)

    return voronoi_xyzs
