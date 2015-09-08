#------------------------------------------------------------------------------
# filename  : make_cs_grid.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.2      srart
#             2014.4.22     redefine the structure data
#             2014.5.26     add make_cs_grid_coords_netcdf()
#             2015.9.4      refactoring
#
#
# description: 
#   Generate the grid and coordinates of the cubed-sphere grid
#
# subroutines:
#   make_cs_grid_netcdf()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4

from pkg.util.quadrature import gausslobatto
from pkg.convert_coord.cs_ll import abp2latlon



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




def ij2ab(ne, ngq, gi, gj, ei, ej, panel):
    gq_pts, gq_wts = gausslobatto(ngq-1)
    delta_angles = (gq_pts[:] + 1)*np.pi/(4*ne) 

    alpha = -np.pi/4 + np.pi/(2*ne)*(ei-1) + delta_angles[gi-1]
    beta  = -np.pi/4 + np.pi/(2*ne)*(ej-1) + delta_angles[gj-1]

    return alpha, beta




def get_across_ij_panel_side(ne, ngq, tag1, tag2, gi, gj, ei, ej):
    p1, ewsn1 = int(tag1[0]), tag1[1]
    p2, ewsn2 = int(tag2[0]), tag2[1]

    if ewsn2 == 'W':
        gi2, ei2 = 1, 1

        if p2 in [1,2,3,4]:
            gj2, ej2 = gj, ej

        elif p2 == 5:
            gj2, ej2 = gi, ei

        elif p2 == 6:
            gj2, ej2 = ngq-gi+1, ne-ei+1

    elif ewsn2 == 'E':
        gi2, ei2 = ngq, ne

        if p2 in [1,2,3,4]:
            gj2, ej2 = gj, ej

        elif p2 == 5:
            gj2, ej2 = ngq-gi+1, ne-ei+1

        elif p2 == 6:
            gj2, ej2 = gi, ei

    elif ewsn2 == 'S':
        gj2, ej2 = 1, 1

        if p2 in [1,6]:
            gi2, ei2 = gi, ei

        elif p2 == 2:
            gi2, ei2 = ngq-gj+1, ne-ej+1

        elif p2 in [3,5]:
            gi2, ei2 = ngq-gi+1, ne-ei+1

        elif p2 == 4:
            gi2, ei2 = gj, ej

    elif ewsn2 == 'N':
        gj2, ej2 = ngq, ne

        if p2 in [1,5]:
            gi2, ei2 = gi, ei

        elif p2 == 2:
            gi2, ei2 = gj, ej

        elif p2 in [3,6]:
            gi2, ei2 = ngq-gi+1, ne-ei+1

        elif p2 == 4:
            gi2, ei2 = ngq-gj+1, ne-ej+1


    return (gi2,gj2,ei2,ej2,p2)




def get_next_ij_panel_side(ewsn, ngq, gi, gj, ei, ej, panel):
    if ewsn == 'W':
        if gj == 1:
            return 'backward', (gi,ngq,ei,ej-1,panel)
        else:
            return 'forward',  (gi,  1,ei,ej+1,panel)

    elif ewsn == 'E':
        if gj == 1:
            return 'forward',  (gi,ngq,ei,ej-1,panel)
        else:
            return 'backward', (gi,  1,ei,ej+1,panel)

    elif ewsn == 'S':
        if gi == 1:
            return 'forward',  (ngq,gj,ei-1,ej,panel)
        else:
            return 'backward', (  1,gj,ei+1,ej,panel)

    elif ewsn == 'N':
        if gi == 1:
            return 'backward', (ngq,gj,ei-1,ej,panel)
        else:
            return 'forward',  (  1,gj,ei+1,ej,panel)




def get_neighbors(ngq, ij2seq_dict, gi, gj, ei, ej, panel):
    if (gi,gj) in [(1,1),(ngq,1),(ngq,ngq),(1,ngq)]:
        if (gi,gj) == (1,1):
            nb0 = ij2seq_dict[(2,1,ei,ej,panel)]
            nb1 = ij2seq_dict[(2,2,ei,ej,panel)]
            nb2 = ij2seq_dict[(1,2,ei,ej,panel)]

        elif (gi,gj) == (ngq,1):
            nb0 = ij2seq_dict[(ngq  ,2,ei,ej,panel)]
            nb1 = ij2seq_dict[(ngq-1,2,ei,ej,panel)]
            nb2 = ij2seq_dict[(ngq-1,1,ei,ej,panel)]

        elif (gi,gj) == (ngq,ngq):
            nb0 = ij2seq_dict[(ngq-1,ngq  ,ei,ej,panel)]
            nb1 = ij2seq_dict[(ngq-1,ngq-1,ei,ej,panel)]
            nb2 = ij2seq_dict[(ngq  ,ngq-1,ei,ej,panel)]

        elif (gi,gj) == (1,ngq):
            nb0 = ij2seq_dict[(1,ngq-1,ei,ej,panel)]
            nb1 = ij2seq_dict[(2,ngq-1,ei,ej,panel)]
            nb2 = ij2seq_dict[(2,ngq  ,ei,ej,panel)]

        return nb0, nb1, nb2

    elif (gi in [1,ngq]) or (gj in [1,ngq]):
        if gi == 1:
            nb0 = ij2seq_dict[(1,gj-1,ei,ej,panel)]
            nb1 = ij2seq_dict[(2,gj-1,ei,ej,panel)]
            nb2 = ij2seq_dict[(2,gj  ,ei,ej,panel)]
            nb3 = ij2seq_dict[(2,gj+1,ei,ej,panel)]
            nb4 = ij2seq_dict[(1,gj+1,ei,ej,panel)]

        elif gi == ngq:
            nb0 = ij2seq_dict[(ngq  ,gj+1,ei,ej,panel)]
            nb1 = ij2seq_dict[(ngq-1,gj+1,ei,ej,panel)]
            nb2 = ij2seq_dict[(ngq-1,gj  ,ei,ej,panel)]
            nb3 = ij2seq_dict[(ngq-1,gj-1,ei,ej,panel)]
            nb4 = ij2seq_dict[(ngq  ,gj-1,ei,ej,panel)]

        elif gj == 1:
            nb0 = ij2seq_dict[(gi+1,1,ei,ej,panel)]
            nb1 = ij2seq_dict[(gi+1,2,ei,ej,panel)]
            nb2 = ij2seq_dict[(gi  ,2,ei,ej,panel)]
            nb3 = ij2seq_dict[(gi-1,2,ei,ej,panel)]
            nb4 = ij2seq_dict[(gi-1,1,ei,ej,panel)]

        elif gj == ngq:
            nb0 = ij2seq_dict[(gi-1,ngq  ,ei,ej,panel)]
            nb1 = ij2seq_dict[(gi-1,ngq-1,ei,ej,panel)]
            nb2 = ij2seq_dict[(gi  ,ngq-1,ei,ej,panel)]
            nb3 = ij2seq_dict[(gi+1,ngq-1,ei,ej,panel)]
            nb4 = ij2seq_dict[(gi+1,ngq  ,ei,ej,panel)]

        return nb0, nb1, nb2, nb3, nb4

    else:
        nb0 = ij2seq_dict[(gi-1,gj-1,ei,ej,panel)]
        nb1 = ij2seq_dict[(gi  ,gj-1,ei,ej,panel)]
        nb2 = ij2seq_dict[(gi+1,gj-1,ei,ej,panel)]
        nb3 = ij2seq_dict[(gi+1,gj  ,ei,ej,panel)]
        nb4 = ij2seq_dict[(gi+1,gj+1,ei,ej,panel)]
        nb5 = ij2seq_dict[(gi  ,gj+1,ei,ej,panel)]
        nb6 = ij2seq_dict[(gi-1,gj+1,ei,ej,panel)]
        nb7 = ij2seq_dict[(gi-1,gj  ,ei,ej,panel)]

        return nb0, nb1, nb2, nb3, nb4, nb5, nb6, nb7




def make_cs_grid_coords_netcdf(ne, ngq, rotated=False):
    size = ngq*ngq*ne*ne*6
    uvp_size = 4*(ngq-1)*ne*((ngq-1)*ne + 1) + \
               2*((ngq-1)*(ne-1) + (ngq-2))**2

    # Gauss-qudrature point index (gi,gj,ei,ej,panel)
    gq_indices = np.zeros((size,5), 'i4')

    # sequential index
    mvps = np.ones((size,4), 'i4')*(-1) # overlapped points (gid or -1)
    is_uvps = np.zeros(size, 'i2')      # unique-point (True/False)
    uids = np.ones(size, 'i4')*(-1)     # unique index (seq)
    gids = np.ones(uvp_size, 'i4')*(-1) # global index
    nbrs = np.ones((uvp_size,8), 'i4')*(-1) # neighbors, anti-clockwise (gid)

    # coordinates
    alpha_betas = np.zeros((uvp_size,2), 'f8')
    latlons = np.zeros((uvp_size,2), 'f8')


    #-----------------------------------------------------
    # (gi, gj, ei, ej, panel)
    #-----------------------------------------------------
    seq = 0
    ij2seq_dict = dict()

    for panel in xrange(1,7):
        for ej in xrange(1,ne+1):
            for ei in xrange(1,ne+1):
                for gj in xrange(1,ngq+1):
                    for gi in xrange(1,ngq+1):
                        ij = (gi,gj,ei,ej,panel)
                        gq_indices[seq,:] = ij
                        ij2seq_dict[ij] = seq       # start from 0

                        seq += 1

    
    #-----------------------------------------------------
    # mvps (overlapped index)
    print 'Generate mvps (multi-valued points)'
    #-----------------------------------------------------
    abcd2ij_dict = {'A':(1,1,1,1), 'B':(ngq,1,ne,1), \
                    'C':(ngq,ngq,ne,ne), 'D':(1,ngq,1,ne)}

    for seq in xrange(size):
        gi, gj, ei, ej, panel = gq_indices[seq]

        mvps[seq,0] = seq       # self index, start from 0

        #-----------------------------------
        # At the panel corner (3 points)
        #-----------------------------------
        if (gi,gj,ei,ej) in abcd2ij_dict.values():
            if (gi,gj,ei,ej) == (1,1,1,1):         tag1 = '%dA'%(panel)
            elif (gi,gj,ei,ej) == (ngq,1,ne,1):    tag1 = '%dB'%(panel)
            elif (gi,gj,ei,ej) == (ngq,ngq,ne,ne): tag1 = '%dC'%(panel)
            elif (gi,gj,ei,ej) == (1,ngq,1,ne):    tag1 = '%dD'%(panel)

            tag2 = panel_corner_tags[tag1]
            tag3 = panel_corner_tags[tag2]

            for k, tag in enumerate([tag2,tag3]):
                p, abcd = int(tag[0]), tag[1]
                ij = tuple( list(abcd2ij_dict[abcd]) + [p] )
                mvps[seq,k+1] = ij2seq_dict[ij]

            #print seq, mvps[seq,:]


        #-----------------------------------
        # At the panel side
        #-----------------------------------
        elif (gi==1 and ei==1) or (gi==ngq and ei==ne) or \
             (gj==1 and ej==1) or (gj==ngq and ej==ne):

            if   gi==1   and ei==1:  tag1 = '%d%s'%(panel, 'W')
            elif gi==ngq and ei==ne: tag1 = '%d%s'%(panel, 'E')
            elif gj==1   and ej==1:  tag1 = '%d%s'%(panel, 'S')
            elif gj==ngq and ej==ne: tag1 = '%d%s'%(panel, 'N')

            tag2 = panel_side_tags[tag1]
            p1, ewsn1 = int(tag1[0]), tag1[1]
            p2, ewsn2 = int(tag2[0]), tag2[1]

            ij0 = (gi,gj,ei,ej,p1)
            ij_across = get_across_ij_panel_side( \
                    ne, ngq, tag1, tag2, gi, gj, ei, ej)

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
                ij1 = (ngq,  1,ei-1,ej  ,panel)
                ij2 = (ngq,ngq,ei-1,ej-1,panel)
                ij3 = (  1,ngq,ei  ,ej-1,panel)

            elif (gi,gj) == (ngq,1):
                ij1 = (ngq,ngq,ei  ,ej-1,panel)
                ij2 = (  1,ngq,ei+1,ej-1,panel)
                ij3 = (  1,  1,ei+1,ej  ,panel)

            elif (gi,gj) == (ngq,ngq):
                ij1 = (  1,ngq,ei+1,ej  ,panel)
                ij2 = (  1,  1,ei+1,ej+1,panel)
                ij3 = (ngq,  1,ei  ,ej+1,panel)

            elif (gi,gj) == (1,ngq):
                ij1 = (  1,  1,ei  ,ej+1,panel)
                ij2 = (ngq,  1,ei-1,ej+1,panel)
                ij3 = (ngq,ngq,ei-1,ej  ,panel)

            #print '(gi,gj,ei,ej,panel)', (gi,gj,ei,ej,panel)
            mvps[seq,1] = ij2seq_dict[ij1]
            mvps[seq,2] = ij2seq_dict[ij2]
            mvps[seq,3] = ij2seq_dict[ij3]


        #-----------------------------------
        # 2 points inside the panel
        #-----------------------------------
        #  side
        elif gi in (1,ngq) or gj in (1,ngq):
            if   gj == 1:   ij1 = (gi,ngq,ei,ej-1,panel)
            elif gj == ngq: ij1 = (gi,  1,ei,ej+1,panel)
            elif gi == 1:   ij1 = (ngq,gj,ei-1,ej,panel)
            elif gi == ngq: ij1 = (  1,gj,ei+1,ej,panel)

            mvps[seq,1] = ij2seq_dict[ij1]


    #-----------------------------------------------------
    # is_uvps (unique-point, True/False)
    # uids (unique index), size
    # gids (global index), uvp_size
    print 'Generate is_uvps, uids and gids'
    #-----------------------------------------------------
    u_seq = 0

    for seq in xrange(size):
        valid_mvp = [k for k in mvps[seq] if k != -1]

        #print seq, valid_mvp
        if min(valid_mvp) == seq:
            is_uvps[seq] = True
            gids[u_seq] = seq

            for k in valid_mvp:
                uids[k] = u_seq

            u_seq += 1

    is_uvp_size = np.count_nonzero(is_uvps)
    assert uvp_size == is_uvp_size, 'Error: uvp_size=%d, np.count_nonzero(is_uvp)=%d'%(uvp_size, is_uvp_size)
    assert uvp_size == u_seq, 'Error: uvp_size=%d, u_seq=%d'%(uvp_size, u_seq)
    assert -1 not in uids, 'Error: -1 in uids'
    assert -1 not in gids, 'Error: -1 in gids'



    #-----------------------------------------------------
    # nbrs (neighbors)
    print 'Generate nbrs (neighbors, anti-clockwise)'
    #-----------------------------------------------------
    for u_seq in xrange(uvp_size):
        gid = gids[u_seq]
        gi, gj, ei, ej, panel = gq_indices[gid]
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
    # coordinates  (alpha,beta), (lat,lon)
    print 'Generate coordinates (alpha,beta), (lat,lon)'
    #-----------------------------------------------------
    for u_seq in xrange(uvp_size):
        seq = gids[u_seq]
        gi, gj, ei, ej, panel = gq_indices[seq]

        alpha, beta = ij2ab(ne, ngq, gi, gj, ei, ej, panel)
        alpha_betas[u_seq,:] = (alpha, beta)

        if rotated:
            rlat, rlon = np.deg2rad(38), np.deg2rad(127.5)  # korea centered
        else:
            rlat, rlon = 0, 0
        lat, lon = abp2latlon(alpha, beta, panel, rlat, rlon)
        latlons[u_seq,:] = (lat,lon)



    #-----------------------------------------------------
    # Save as NetCDF format
    print 'Save as NetCDF'
    #-----------------------------------------------------
    ncf = netCDF4.Dataset('cs_grid_ne%dngq%d.nc'%(ne,ngq), 'w', format='NETCDF4')
    ncf.description = 'Cubed-Sphere grid coordinates'
    ncf.notice = 'All sequential indices start from 0 except for the gq_indices'
    ncf.rotated = np.int8(rotated)
    ncf.createDimension('ne', ne)
    ncf.createDimension('ngq', ngq)
    ncf.createDimension('size', size)
    ncf.createDimension('uvp_size', uvp_size)
    ncf.createDimension('2', 2)
    ncf.createDimension('4', 4)
    ncf.createDimension('5', 5)
    ncf.createDimension('8', 8)

    vgq_indices = ncf.createVariable('gq_indices', 'i4', ('size','5'))
    vgq_indices.long_name = 'Gauss-quadrature point indices, (gi,gj,ei,ej,panel)'
    vgq_indices.units = 'index'

    vmvps = ncf.createVariable('mvps', 'i4', ('size','4'))
    vmvps.long_name = 'Indices of the Multiple-Value Points'
    vmvps.units = 'index'
    vis_uvps = ncf.createVariable('is_uvps', 'i2', ('size',))
    vis_uvps.long_name = 'Is this a Unique-Value Point?'
    vis_uvps.units = 'boolean'
    vuids = ncf.createVariable('uids', 'i4', ('size',))
    vuids.long_name = 'Unique-point sequence'
    vuids.units = 'index'
    vgids = ncf.createVariable('gids', 'i4', ('uvp_size',))
    vgids.long_name = 'Global-point sequence'
    vgids.units = 'index'
    vnbrs = ncf.createVariable('nbrs', 'i4', ('uvp_size','8'))
    vnbrs.long_name = 'Neighbors, anti-clockwise direction with gid'
    vnbrs.units = 'index'

    valpha_betas = ncf.createVariable('alpha_betas', 'f8', ('uvp_size','2'))
    valpha_betas.long_name = '(alpha,beta), angle in a panel'
    valpha_betas.units = 'radian'
    vlatlons = ncf.createVariable('latlons', 'f8', ('uvp_size','2'))
    vlatlons.long_name = '(lat,lon)'
    vlatlons.units = 'radians_north, radians_east'


    vgq_indices[:] = gq_indices[:]
    vmvps[:] = mvps[:]
    vis_uvps[:] = is_uvps[:]
    vuids[:] = uids[:]
    vgids[:] = gids[:]
    vnbrs[:] = nbrs[:]
    valpha_betas[:] = alpha_betas[:]
    vlatlons[:] = latlons[:]

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

    make_cs_grid_coords_netcdf(args.ne, args.ngq, args.rotated)
