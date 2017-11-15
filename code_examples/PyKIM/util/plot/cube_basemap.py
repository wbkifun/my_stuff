#------------------------------------------------------------------------------
# filename  : cube_basemap.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2014.4.26     start
#             2014.5.7      add linetype='cubeline'
#             2015.12.17    revision
#
#
# description: 
#   Plot on the sphere using basemap
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy import pi, sqrt, rad2deg, deg2rad, arange
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

from util.convert_coord.cs_ll import xyp2latlon, abp2latlon
from util.geometry.greatcircle import GreatCircle
from util.geometry.circumcircle import Circumcircle



at = 1/np.sqrt(3)*np.tan(np.pi/4)       # R=1, at=0.577
panel_corner_xys = {'A':(-at,-at), 'B':(at,-at), 'C':(at,at), 'D':(-at,at)}
panel_side2corners = \
        {'S':('A','B'), 'E':('B','C'), 'N':('C','D'), 'W':('D','A')}

PPD = 1         # ppd : points per degree (resolution)




def make_greatcircle_points(latlon1, latlon2, ppd=PPD):
    gc = GreatCircle(latlon1, latlon2)
    npts = max(2, int(gc.max_angle*360*ppd/(2*pi)))

    latlons = list()
    for phi in np.linspace(0, gc.max_angle, npts):
        latlons.append( gc.phi2latlon(phi) )

    return latlons




def draw_line(bmap, latlon1, latlon2, ppd=PPD, **kargs):
    kwds = dict(c='k', lw=1., ls='-')           # default parameters
    kwds.update(kargs)

    latlons = make_greatcircle_points(latlon1, latlon2, ppd)
    lls = np.asarray(latlons)
    xs, ys = bmap(rad2deg(lls[:,1]), rad2deg(lls[:,0]))
    line, = bmap.plot(xs[xs<1e10], ys[ys<1e10], **kwds)

    return line




def draw_polygon(bmap, polygon_latlons, ppd=PPD, **kargs):
    kwds = dict(fill=True, fc='w', ec='k', alpha=0.5)
    kwds.update(kargs)

    pls = polygon_latlons
    latlons = list()
    for latlon1, latlon2 in zip(pls, pls[1:]+pls[:-1]):
        latlons.extend( make_greatcircle_points(latlon1, latlon2, ppd) )

    lls = np.asarray(latlons)
    xs, ys = bmap(rad2deg(lls[:,1]), rad2deg(lls[:,0]))
    xys = zip(xs[xs<1e10], ys[ys<1e10])
    poly = plt.gca().add_patch( Polygon(xys, **kwds) )

    return poly




def draw_points(bmap, latlons, **kargs):
    kwds = dict(s=20, c='k', marker='o', alpha=0.5)
    kwds.update(kargs)

    lls = np.asarray(latlons)
    xs, ys = bmap(rad2deg(lls[:,1]), rad2deg(lls[:,0]))
    bmap.scatter(xs, ys, **kwds)




def draw_circumcircle(bmap, center_latlon, latlon, res=100, **kargs):
    kwds = dict(fill=True, fc='y', ec='r', lw=1, alpha=0.5)
    kwds.update(kargs)

    cc = Circumcircle(center_latlon, latlon)

    latlons = list()
    for phi in np.linspace(0,2*pi,res):
        latlons.append( cc.phi2latlon(phi) )

    lls = np.asarray(latlons)
    xs, ys = bmap(rad2deg(lls[:,1]), rad2deg(lls[:,0]))
    xys = zip(xs[xs<1e10], ys[ys<1e10])
    poly = plt.gca().add_patch( Polygon(xys, **kwds) )

    # center point
    xs, ys = bmap(rad2deg(center_latlon[1]), rad2deg(center_latlon[0]))
    bmap.scatter(xs, ys, s=20, c='r', marker='o', alpha=1)

    return poly




class PlotSphere(object):
    def __init__(self, lat0, lon0, figsize=(12,12), interact=True, draw_map=True, rotate=True):
        if interact:
            plt.ion()
        else:
            plt.ioff()

        if rotate:
            self.rlat, self.rlon = deg2rad(38), deg2rad(127)
        else:
            self.rlat, self.rlon = 0, 0

        fig = plt.figure(figsize=figsize)

        #-------------------------------------------------------------
        # orthographic projection
        # lon_0, lat_0 are the center point of the projection
        # resolution = 'c' means use 'crude' resolution coastlines
        #-------------------------------------------------------------
        self.bmap = Basemap(projection='ortho', resolution='c', \
                            lon_0=lon0, lat_0=lat0)

        if draw_map:
            self.bmap.drawcoastlines(linewidth=0.2)
            self.bmap.fillcontinents(color='coral', lake_color='aqua')
            self.bmap.drawmapboundary(fill_color='aqua')
            plt.tight_layout(pad=1)

        self.plt = plt
        self.savefig = plt.savefig
        self.draw = plt.draw
        self.show = plt.show



    def save_png_eps(self, fname, dpi=120):
        plt.savefig(fname+'.png')
        plt.savefig(fname+'.eps')



    def draw_cube_panel(self, panel, **kargs):
        kwds = dict(c='0.5', lw=1., ls='-')
        kwds.update(kargs)

        for side, (corner1,corner2) in panel_side2corners.items():
            xy1, xy2 = panel_corner_xys[corner1], panel_corner_xys[corner2]

            latlon1 = xyp2latlon(xy1[0], xy1[1], panel, self.rlat, self.rlon)
            latlon2 = xyp2latlon(xy2[0], xy2[1], panel, self.rlat, self.rlon)
            draw_line(self.bmap, latlon1, latlon2, **kwds)


    def draw_cube_elements(self, ne, panel, **kargs):
        kwds = dict(c='0.5', lw=1., ls='-')
        kwds.update(kargs)

        angles = np.linspace(-pi/4,pi/4,ne+1)

        for angle in angles:
            latlon1 = abp2latlon(angle,-pi/4,panel, self.rlat, self.rlon)
            latlon2 = abp2latlon(angle,pi/4,panel, self.rlat, self.rlon)
            draw_line(self.bmap, latlon1, latlon2, **kwds)

        for angle in angles:
            latlon1 = abp2latlon(-pi/4,angle,panel, self.rlat, self.rlon)
            latlon2 = abp2latlon(pi/4,angle,panel, self.rlat, self.rlon)
            draw_line(self.bmap, latlon1, latlon2, **kwds)


    def draw_latlon(self):
        self.bmap.drawparallels(arange(-90,101,10))
        self.bmap.drawmeridians(arange(0,360,10))




if __name__ == '__main__':
    ps = PlotSphere(38, 127, figsize=(12,12))     # Korea centered

    '''
    for panel in range(1,7):
        ps.draw_cube_elements(10, panel, c='0.3')

    for panel in range(1,7):
        ps.draw_cube_panel(panel, c='k')
    '''
    ps.draw_latlon()

    #latlon1 = xyp2latlon(0.5, 0.4, 2)
    #latlon2 = xyp2latlon(0.4, 0.5, 6)

    #latlon1 = xyp2latlon(at, 0.4, 2)
    #latlon2 = xyp2latlon(0.4, at, 6)

    #latlon1 = xyp2latlon(0.4, 0.3, 2)
    #latlon2 = xyp2latlon(-0.4, 0.2, 3)


    #latlon1 = xyp2latlon(0.5, 0.4, 2)
    #latlon2 = xyp2latlon(0.4, 0.5, 6)
    #latlon3 = xyp2latlon(0.35, 0.5, 6)
    #latlon4 = xyp2latlon(0.5, 0.35, 2)

    #latlon1 = xyp2latlon(0.5, 0.5, 2)
    #latlon2 = xyp2latlon(0.35, 0.5, 6)
    #latlon3 = xyp2latlon(at, at, 6)
    #latlon4 = xyp2latlon(0.5, 0.35, 2)
    #draw_greatcircle_line(ps.bmap, latlon1, latlon2, c='r')
    #draw_greatcircle_line(ps.bmap, latlon3, latlon4, c='b')

    #latlons = [latlon1, latlon4, latlon3, latlon2] 
    #poly = draw_polygon(ps.bmap, latlons)
    #poly.update(dict(fc='r'))

    #draw_points(ps.bmap, latlons)

    #ps.save_png_eps('cube_ne10_rotated', dpi=120)
    ps.save_png_eps('latlon', dpi=120)
    ps.show(True)
