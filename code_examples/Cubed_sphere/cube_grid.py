#------------------------------------------------------------------------------
# filename  : cube_grid.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.2      srart
#             2014.4.22     redefine the structure data
#             2014.5.26     add make_cs_grid_coords_netcdf()
#             2015.9.4      refactoring
#             2015.9.8      change (gi,gj,ei,ej,panel) -> (panel,ei,ej,gi,gj)
#             2015.9.11     change to class
#             2015.10.12    append some netcdf global attributes
#
#
# description: 
#   Generate the grid and coordinates of the cubed-sphere grid
#
# class:
#   CubedSphereGrid()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from datetime import datetime

from pkg.util.quadrature import gausslobatto
from pkg.convert_coord.cs_ll import abp2latlon
from pkg.convert_coord.cart_ll import latlon2xyz



# <neighbor panel side>
#
#          ----- 
#         |  N  |
#         |W 6 E|
#         |  S  |
#          ----- 
#  -----   -----   -----   -----
# |  N  | |  N  | |  N  | |  N  |
# |W 4 E| |W 1 E| |W 2 E| |W 3 E|
# |  S  | |  S  | |  S  | |  S  |
#  -----   -----   -----   -----
#          -----  
#         |  N  |
#         |W 5 E|
#         |  S  |
#          ----- 

panel_side_tags = \
        {'1S':'5N', '5N':'1S', '1E':'2W', '2W':'1E',
         '1N':'6S', '6S':'1N', '1W':'4E', '4E':'1W',
         '3S':'5S', '5S':'3S', '3E':'4W', '4W':'3E',
         '3N':'6N', '6N':'3N', '3W':'2E', '2E':'3W',
         '2S':'5E', '5E':'2S', '2N':'6E', '6E':'2N',
         '4S':'5W', '5W':'4S', '4N':'6W', '6W':'4N'}



# <neighbor panel corner>
#
#          ----- 
#         |D   C|
#         |  6  |
#         |A   B|
#          ----- 
#  -----   -----   -----   -----
# |D   C| |D   C| |D   C| |D   C|
# |  4  | |  1  | |  2  | |  3  |
# |A   B| |A   B| |A   B| |A   B|
#  -----   -----   -----   -----
#          -----  
#         |D   C|
#         |  5  |
#         |A   B|
#          ----- 

panel_corner_tags = \
        {'1A':'4B', '4B':'5D', '5D':'1A',
         '1B':'5C', '5C':'2A', '2A':'1B',
         '1C':'2D', '2D':'6B', '6B':'1C',
         '1D':'6A', '6A':'4C', '4C':'1D',
         '3A':'2B', '2B':'5B', '5B':'3A',
         '3B':'5A', '5A':'4A', '4A':'3B',
         '3C':'4D', '4D':'6D', '6D':'3C',
         '3D':'6C', '6C':'2C', '2C':'3D'}


at = 1/np.sqrt(3)*np.tan(np.pi/4)       # R=1, at=0.577
panel_corner_xys = {'A':(-at,-at), 'B':(at,-at), 'C':(at,at), 'D':(-at,at)}




def ij2ab(ne, ngq, panel, ei, ej, gi, gj):
    gq_pts, gq_wts = gausslobatto(ngq-1)
    delta_angles = (gq_pts[:] + 1)*np.pi/(4*ne) 

    alpha = -np.pi/4 + np.pi/(2*ne)*(ei-1) + delta_angles[gi-1]
    beta  = -np.pi/4 + np.pi/(2*ne)*(ej-1) + delta_angles[gj-1]

    return alpha, beta




def get_across_ij_panel_side(ne, ngq, tag1, tag2, ei, ej, gi, gj):
    p1, ewsn1 = int(tag1[0]), tag1[1]
    p2, ewsn2 = int(tag2[0]), tag2[1]

    if ewsn2 == 'W':
        ei2, gi2 = 1, 1

        if p2 in [1,2,3,4]:
            ej2, gj2 = ej, gj

        elif p2 == 5:
            ej2, gj2 = ei, gi

        elif p2 == 6:
            ej2, gj2 = ne-ei+1, ngq-gi+1

    elif ewsn2 == 'E':
        ei2, gi2 = ne, ngq

        if p2 in [1,2,3,4]:
            ej2, gj2 = ej, gj

        elif p2 == 5:
            ej2, gj2 = ne-ei+1, ngq-gi+1

        elif p2 == 6:
            ej2, gj2 = ei, gi

    elif ewsn2 == 'S':
        ej2, gj2 = 1, 1

        if p2 in [1,6]:
            ei2, gi2 = ei, gi

        elif p2 == 2:
            ei2, gi2 = ne-ej+1, ngq-gj+1

        elif p2 in [3,5]:
            ei2, gi2 = ne-ei+1, ngq-gi+1

        elif p2 == 4:
            ei2, gi2 = ej, gj

    elif ewsn2 == 'N':
        ej2, gj2 = ne, ngq

        if p2 in [1,5]:
            ei2, gi2 = ei, gi

        elif p2 == 2:
            ei2, gi2 = ej, gj

        elif p2 in [3,6]:
            ei2, gi2 = ne-ei+1, ngq-gi+1

        elif p2 == 4:
            ei2, gi2 = ne-ej+1, ngq-gj+1


    return (p2,ei2,ej2,gi2,gj2)




def get_next_ij_panel_side(ewsn, ngq, panel, ei, ej, gi, gj):
    if ewsn == 'W':
        if gj == 1:
            return 'backward', (panel,ei,ej-1,gi,ngq)
        else:
            return 'forward',  (panel,ei,ej+1,gi,  1)

    elif ewsn == 'E':
        if gj == 1:
            return 'forward',  (panel,ei,ej-1,gi,ngq)
        else:
            return 'backward', (panel,ei,ej+1,gi,  1)

    elif ewsn == 'S':
        if gi == 1:
            return 'forward',  (panel,ei-1,ej,ngq,gj)
        else:
            return 'backward', (panel,ei+1,ej,  1,gj)

    elif ewsn == 'N':
        if gi == 1:
            return 'backward', (panel,ei-1,ej,ngq,gj)
        else:
            return 'forward',  (panel,ei+1,ej,  1,gj)




def get_neighbors(ngq, ij2seq_dict, panel, ei, ej, gi, gj):
    if (gi,gj) in [(1,1),(ngq,1),(ngq,ngq),(1,ngq)]:
        if (gi,gj) == (1,1):
            nb0 = ij2seq_dict[(panel,ei,ej,2,1)]
            nb1 = ij2seq_dict[(panel,ei,ej,2,2)]
            nb2 = ij2seq_dict[(panel,ei,ej,1,2)]

        elif (gi,gj) == (ngq,1):
            nb0 = ij2seq_dict[(panel,ei,ej,ngq  ,2)]
            nb1 = ij2seq_dict[(panel,ei,ej,ngq-1,2)]
            nb2 = ij2seq_dict[(panel,ei,ej,ngq-1,1)]

        elif (gi,gj) == (ngq,ngq):
            nb0 = ij2seq_dict[(panel,ei,ej,ngq-1,ngq  )]
            nb1 = ij2seq_dict[(panel,ei,ej,ngq-1,ngq-1)]
            nb2 = ij2seq_dict[(panel,ei,ej,ngq  ,ngq-1)]

        elif (gi,gj) == (1,ngq):
            nb0 = ij2seq_dict[(panel,ei,ej,1,ngq-1)]
            nb1 = ij2seq_dict[(panel,ei,ej,2,ngq-1)]
            nb2 = ij2seq_dict[(panel,ei,ej,2,ngq  )]

        return nb0, nb1, nb2

    elif (gi in [1,ngq]) or (gj in [1,ngq]):
        if gi == 1:
            nb0 = ij2seq_dict[(panel,ei,ej,1,gj-1)]
            nb1 = ij2seq_dict[(panel,ei,ej,2,gj-1)]
            nb2 = ij2seq_dict[(panel,ei,ej,2,gj  )]
            nb3 = ij2seq_dict[(panel,ei,ej,2,gj+1)]
            nb4 = ij2seq_dict[(panel,ei,ej,1,gj+1)]

        elif gi == ngq:
            nb0 = ij2seq_dict[(panel,ei,ej,ngq  ,gj+1)]
            nb1 = ij2seq_dict[(panel,ei,ej,ngq-1,gj+1)]
            nb2 = ij2seq_dict[(panel,ei,ej,ngq-1,gj  )]
            nb3 = ij2seq_dict[(panel,ei,ej,ngq-1,gj-1)]
            nb4 = ij2seq_dict[(panel,ei,ej,ngq  ,gj-1)]

        elif gj == 1:
            nb0 = ij2seq_dict[(panel,ei,ej,gi+1,1)]
            nb1 = ij2seq_dict[(panel,ei,ej,gi+1,2)]
            nb2 = ij2seq_dict[(panel,ei,ej,gi  ,2)]
            nb3 = ij2seq_dict[(panel,ei,ej,gi-1,2)]
            nb4 = ij2seq_dict[(panel,ei,ej,gi-1,1)]

        elif gj == ngq:
            nb0 = ij2seq_dict[(panel,ei,ej,gi-1,ngq  )]
            nb1 = ij2seq_dict[(panel,ei,ej,gi-1,ngq-1)]
            nb2 = ij2seq_dict[(panel,ei,ej,gi  ,ngq-1)]
            nb3 = ij2seq_dict[(panel,ei,ej,gi+1,ngq-1)]
            nb4 = ij2seq_dict[(panel,ei,ej,gi+1,ngq  )]

        return nb0, nb1, nb2, nb3, nb4

    else:
        nb0 = ij2seq_dict[(panel,ei,ej,gi-1,gj-1)]
        nb1 = ij2seq_dict[(panel,ei,ej,gi  ,gj-1)]
        nb2 = ij2seq_dict[(panel,ei,ej,gi+1,gj-1)]
        nb3 = ij2seq_dict[(panel,ei,ej,gi+1,gj  )]
        nb4 = ij2seq_dict[(panel,ei,ej,gi+1,gj+1)]
        nb5 = ij2seq_dict[(panel,ei,ej,gi  ,gj+1)]
        nb6 = ij2seq_dict[(panel,ei,ej,gi-1,gj+1)]
        nb7 = ij2seq_dict[(panel,ei,ej,gi-1,gj  )]

        return nb0, nb1, nb2, nb3, nb4, nb5, nb6, nb7




class CubedSphereGrid(object):
    def __init__(self, ne, ngq, rotated=False, is_print=False):
        self.ne = ne
        self.ngq = ngq
        self.rotated = rotated
        self.is_print = is_print

        # EP(Entire Point), UP(Unique Point)
        ep_size = ngq*ngq*ne*ne*6
        up_size = 4*(ngq-1)*ne*((ngq-1)*ne + 1) + \
                   2*((ngq-1)*(ne-1) + (ngq-2))**2


        #-----------------------------------------------------
        # allocations
        #-----------------------------------------------------
        # Gauss-qudrature point index (panel,ei,ej,gi,gj)
        gq_indices = np.zeros((ep_size,5), 'i4')
        ij2seq_dict = dict()    # {(panel,ei,ej,gi,gj):seq,...}

        # sequential index
        mvps = np.ones((ep_size,4), 'i4')*(-1) # overlapped points (gid or -1)
        is_uvps = np.zeros(ep_size, 'i2')      # unique-point (True/False)
        uids = np.ones(ep_size, 'i4')*(-1)     # unique index (seq)
        gids = np.ones(up_size, 'i4')*(-1)     # global index
        nbrs = np.ones((up_size,8), 'i4')*(-1) # neighbors, anti-clockwise (gid)

        # coordinates
        alpha_betas = np.zeros((ep_size,2), 'f8')   # different at each panel
        latlons = np.zeros((up_size,2), 'f8')
        xyzs = np.zeros((up_size,3), 'f8')


        #-----------------------------------------------------
        # (gi, gj, ei, ej, panel)
        #-----------------------------------------------------
        seq = 0
        for panel in xrange(1,7):
            for ej in xrange(1,ne+1):
                for ei in xrange(1,ne+1):
                    for gj in xrange(1,ngq+1):
                        for gi in xrange(1,ngq+1):
                            ij = (panel,ei,ej,gi,gj)
                            gq_indices[seq,:] = ij
                            ij2seq_dict[ij] = seq       # start from 0
                            seq += 1

        
        #-----------------------------------------------------
        # mvps (overlapped index)
        if is_print: print 'Generate mvps (multi-valued points)'
        #-----------------------------------------------------
        abcd2ij_dict = {'A':(1,1,1,1), 'B':(ne,1,ngq,1), \
                        'C':(ne,ne,ngq,ngq), 'D':(1,ne,1,ngq)}

        for seq in xrange(ep_size):
            panel, ei, ej, gi, gj = gq_indices[seq]

            mvps[seq,0] = seq       # self index, start from 0

            #-----------------------------------
            # At the panel corner (3 points)
            #-----------------------------------
            if (ei,ej,gi,gj) in abcd2ij_dict.values():
                if   (ei,ej,gi,gj) == abcd2ij_dict['A']: tag1 = '%dA'%(panel)
                elif (ei,ej,gi,gj) == abcd2ij_dict['B']: tag1 = '%dB'%(panel)
                elif (ei,ej,gi,gj) == abcd2ij_dict['C']: tag1 = '%dC'%(panel)
                elif (ei,ej,gi,gj) == abcd2ij_dict['D']: tag1 = '%dD'%(panel)

                tag2 = panel_corner_tags[tag1]
                tag3 = panel_corner_tags[tag2]

                for k, tag in enumerate([tag2,tag3]):
                    p, abcd = int(tag[0]), tag[1]
                    ij = tuple( [p] + list(abcd2ij_dict[abcd]) )
                    mvps[seq,k+1] = ij2seq_dict[ij]

                #print seq, mvps[seq,:]


            #-----------------------------------
            # At the panel side
            #-----------------------------------
            elif (ei,gi) in [(1,1),(ne,ngq)] or \
                 (ej,gj) in [(1,1),(ne,ngq)]:

                if   (ei,gi) == (1,1):    tag1 = '%d%s'%(panel, 'W')
                elif (ei,gi) == (ne,ngq): tag1 = '%d%s'%(panel, 'E')
                elif (ej,gj) == (1,1):    tag1 = '%d%s'%(panel, 'S')
                elif (ej,gj) == (ne,ngq): tag1 = '%d%s'%(panel, 'N')

                tag2 = panel_side_tags[tag1]
                p1, ewsn1 = int(tag1[0]), tag1[1]
                p2, ewsn2 = int(tag2[0]), tag2[1]

                ij0 = (p1,ei,ej,gi,gj)
                ij_across = get_across_ij_panel_side( \
                        ne, ngq, tag1, tag2, ei, ej, gi, gj)

                #-------------------------------------
                # 4 points
                #-------------------------------------
                if (gi,gj) in [(1,1),(ngq,1),(ngq,ngq),(1,ngq)]:
                    fb_next, ij_next = get_next_ij_panel_side(ewsn1, ngq, *ij0)
                    ij_a_next = get_next_ij_panel_side(ewsn2, ngq, *ij_across)[-1]

                    if fb_next == 'forward':
                        ijs = [ij_next, ij_a_next, ij_across]
                    else:
                        ijs = [ij_across, ij_a_next, ij_next]

                    for k, ij in enumerate(ijs):
                        mvps[seq,k+1] = ij2seq_dict[ij]


                #-------------------------------------
                # 2 points
                #-------------------------------------
                else:
                    mvps[seq,1] = ij2seq_dict[ij_across]


                #print seq, mvps[seq,:]


            #-----------------------------------
            # 4 points inside the panel
            #-----------------------------------
            # corner 
            elif (gi,gj) in [(1,1),(ngq,1),(ngq,ngq),(1,ngq)]:
                if (gi,gj) == (1,1):
                    ij1 = (panel,ei-1,ej  ,ngq,  1)
                    ij2 = (panel,ei-1,ej-1,ngq,ngq)
                    ij3 = (panel,ei  ,ej-1,  1,ngq)

                elif (gi,gj) == (ngq,1):
                    ij1 = (panel,ei  ,ej-1,ngq,ngq)
                    ij2 = (panel,ei+1,ej-1,  1,ngq)
                    ij3 = (panel,ei+1,ej  ,  1,  1)

                elif (gi,gj) == (ngq,ngq):
                    ij1 = (panel,ei+1,ej  ,  1,ngq)
                    ij2 = (panel,ei+1,ej+1,  1,  1)
                    ij3 = (panel,ei  ,ej+1,ngq,  1)

                elif (gi,gj) == (1,ngq):
                    ij1 = (panel,ei  ,ej+1,  1,  1)
                    ij2 = (panel,ei-1,ej+1,ngq,  1)
                    ij3 = (panel,ei-1,ej  ,ngq,ngq)

                #print '(panel,ei,ej,gi,gj)', (panel,ei,ej,gi,gj)
                mvps[seq,1] = ij2seq_dict[ij1]
                mvps[seq,2] = ij2seq_dict[ij2]
                mvps[seq,3] = ij2seq_dict[ij3]


            #-----------------------------------
            # 2 points inside the panel
            #-----------------------------------
            #  side
            elif gi in (1,ngq) or gj in (1,ngq):
                if   gj == 1:   ij1 = (panel,ei,ej-1,gi,ngq)
                elif gj == ngq: ij1 = (panel,ei,ej+1,gi,  1)
                elif gi == 1:   ij1 = (panel,ei-1,ej,ngq,gj)
                elif gi == ngq: ij1 = (panel,ei+1,ej,  1,gj)

                mvps[seq,1] = ij2seq_dict[ij1]


        #-----------------------------------------------------
        # is_uvps (unique-point, True/False)
        # uids (unique index), ep_size
        # gids (global index), up_size
        if is_print: print 'Generate is_uvps, uids and gids'
        #-----------------------------------------------------
        u_seq = 0

        for seq in xrange(ep_size):
            valid_mvp = [k for k in mvps[seq] if k != -1]

            #print seq, valid_mvp
            if min(valid_mvp) == seq:
                is_uvps[seq] = True
                gids[u_seq] = seq

                for k in valid_mvp:
                    uids[k] = u_seq

                u_seq += 1

        is_up_size = np.count_nonzero(is_uvps)
        assert up_size == is_up_size, 'Error: up_size=%d, np.count_nonzero(is_uvp)=%d'%(up_size, is_up_size)
        assert up_size == u_seq, 'Error: up_size=%d, u_seq=%d'%(up_size, u_seq)
        assert -1 not in uids, 'Error: -1 in uids'
        assert -1 not in gids, 'Error: -1 in gids'



        #-----------------------------------------------------
        # nbrs (neighbors)
        if is_print: print 'Generate nbrs (neighbors, anti-clockwise)'
        #-----------------------------------------------------
        for u_seq in xrange(up_size):
            gid = gids[u_seq]
            panel, ei, ej, gi, gj = gq_indices[gid]
            valid_mvp = [k for k in mvps[gid] if k != -1]

            if (gi,gj) in [(1,1),(ngq,1),(ngq,ngq),(1,ngq)]:
                if len(valid_mvp) == 3:     # panel corner
                    ij0, ij1, ij2 = [gq_indices[m] for m in valid_mvp]
                    nbrs[u_seq,:3] = get_neighbors(ngq, ij2seq_dict, *ij0)
                    nbrs[u_seq,3:5] = get_neighbors(ngq, ij2seq_dict, *ij1)[1:]
                    nbrs[u_seq,5] = get_neighbors(ngq, ij2seq_dict, *ij2)[1]

                elif len(valid_mvp) == 4:   # corner
                    ij0, ij1, ij2, ij3 = [gq_indices[m] for m in valid_mvp]
                    nbrs[u_seq,:3] = get_neighbors(ngq, ij2seq_dict, *ij0)
                    nbrs[u_seq,3:5] = get_neighbors(ngq, ij2seq_dict, *ij1)[1:]
                    nbrs[u_seq,5:7] = get_neighbors(ngq, ij2seq_dict, *ij2)[1:]
                    nbrs[u_seq,7] = get_neighbors(ngq, ij2seq_dict, *ij3)[1]


            elif (gi in [1,ngq]) or (gj in [1,ngq]):
                ij0, ij1 = [gq_indices[m] for m in valid_mvp]
                nbrs[u_seq,:5] = get_neighbors(ngq, ij2seq_dict, *ij0)
                nbrs[u_seq,5:] = get_neighbors(ngq, ij2seq_dict, *ij1)[1:-1]


            else:
                ij0 = gq_indices[valid_mvp[0]]
                nbrs[u_seq,:] = get_neighbors(ngq, ij2seq_dict, *ij0)

            #print u_seq, gid, nbrs[u_seq]



        #-----------------------------------------------------
        # coordinates  (alpha,beta), (lat,lon), (x,y,z)
        if is_print: print 'Generate coordinates (alpha,beta), (lat,lon), (x,y,z)'
        #-----------------------------------------------------
        for seq in xrange(ep_size):
            panel, ei, ej, gi, gj = gq_indices[seq]
            alpha, beta = ij2ab(ne, ngq, panel, ei, ej, gi, gj)
            alpha_betas[seq,:] = (alpha, beta)


        for u_seq in xrange(up_size):
            seq = gids[u_seq]
            panel, ei, ej, gi, gj = gq_indices[seq]

            alpha, beta = alpha_betas[seq]

            if rotated:
                rlat, rlon = np.deg2rad(38), np.deg2rad(127.5)  #korea centered
            else:
                rlat, rlon = 0, 0
            lat, lon = abp2latlon(alpha, beta, panel, rlat, rlon)
            latlons[u_seq,:] = (lat,lon)

            x, y, z = latlon2xyz(lat, lon)
            xyzs[u_seq,:] = (x, y, z)


        #-----------------------------------------------------
        # global variables
        #-----------------------------------------------------
        self.ep_size = ep_size
        self.up_size = up_size
        self.ij2seq_dict = ij2seq_dict
        self.gq_indices = gq_indices
        self.mvps = mvps
        self.is_uvps = is_uvps
        self.uids = uids
        self.gids = gids
        self.nbrs = nbrs
        self.alpha_betas = alpha_betas
        self.latlons = latlons
        self.xyzs = xyzs
        self.rlat = rlat
        self.rlon = rlon



    def save_netcdf(self):
        #-----------------------------------------------------
        # Save as NetCDF format
        if self.is_print: print 'Save as NetCDF'
        #-----------------------------------------------------
        ne, ngq = self.ne, self.ngq

        ncf = nc.Dataset('cs_grid_ne%dngq%d.nc'%(ne,ngq), 'w', format='NETCDF4')
        ncf.description = 'Cubed-Sphere grid coordinates'
        ncf.notice = 'All sequential indices start from 0 except for the gq_indices'
        ncf.ne = np.int32(ne)
        ncf.ngq = np.int32(ngq)
        ncf.ep_size = np.int32(self.ep_size)
        ncf.up_size = np.int32(self.up_size)
        ncf.rotated = np.int8(self.rotated)
        ncf.rlat = self.rlat
        ncf.rlon = self.rlon
        ncf.date_of_production = '%s'%datetime.now()
        ncf.author = 'kh.kim@kiaps.org'

        ncf.createDimension('ep_size', self.ep_size)
        ncf.createDimension('up_size', self.up_size)
        ncf.createDimension('2', 2)
        ncf.createDimension('3', 3)
        ncf.createDimension('4', 4)
        ncf.createDimension('5', 5)
        ncf.createDimension('8', 8)

        vgq_indices = ncf.createVariable('gq_indices', 'i4', ('ep_size','5'))
        vgq_indices.long_name = 'Gauss-quadrature point indices, (panel,ei,ej,gi,gj)'
        vgq_indices.units = 'index'

        vmvps = ncf.createVariable('mvps', 'i4', ('ep_size','4'))
        vmvps.long_name = 'Indices of the Multiple-Value Points'
        vmvps.units = 'index'
        vis_uvps = ncf.createVariable('is_uvps', 'i2', ('ep_size',))
        vis_uvps.long_name = 'Is this a Unique-Value Point?'
        vis_uvps.units = 'boolean'
        vuids = ncf.createVariable('uids', 'i4', ('ep_size',))
        vuids.long_name = 'Unique-point sequence'
        vuids.units = 'index'
        vgids = ncf.createVariable('gids', 'i4', ('up_size',))
        vgids.long_name = 'Global-point sequence'
        vgids.units = 'index'
        vnbrs = ncf.createVariable('nbrs', 'i4', ('up_size','8'))
        vnbrs.long_name = 'Neighbors, anti-clockwise direction with gid'
        vnbrs.units = 'index'

        valpha_betas = ncf.createVariable('alpha_betas', 'f8', ('ep_size','2'))
        valpha_betas.long_name = '(alpha,beta), angle in a panel'
        valpha_betas.units = 'radian'
        vlatlons = ncf.createVariable('latlons', 'f8', ('up_size','2'))
        vlatlons.long_name = '(lat,lon)'
        vlatlons.units = 'radians_north, radians_east'
        vxyzs = ncf.createVariable('xyzs', 'f8', ('up_size','3'))
        vxyzs.long_name = '(x,y,z), Cartesian coordinates'
        vxyzs.units = 'unit radius r=1'


        vgq_indices[:]  = self.gq_indices[:]
        vmvps[:]        = self.mvps[:]
        vis_uvps[:]     = self.is_uvps[:]
        vuids[:]        = self.uids[:]
        vgids[:]        = self.gids[:]
        vnbrs[:]        = self.nbrs[:]
        valpha_betas[:] = self.alpha_betas[:]
        vlatlons[:]     = self.latlons[:]
        vxyzs[:]        = self.xyzs[:]

        ncf.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('ngq', type=int, help='number of Gauss qudrature points')
    parser.add_argument('--rotated', action='store_true', help='Korea centered rotation')
    #parser.add_argument('ngq', nargs='?', type=int, default=4, help='number of Gauss qudrature points')
    args = parser.parse_args()

    print 'Generate the information of Cubed-sphere grid'
    print 'ne=%d, ngq=%d, rotated=%s'%(args.ne, args.ngq, args.rotated)

    csgrid = CubedSphereGrid(args.ne, args.ngq, args.rotated, is_print=True)
    csgrid.save_netcdf()
