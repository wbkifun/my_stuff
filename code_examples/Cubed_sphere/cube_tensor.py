#------------------------------------------------------------------------------
# filename  : cube_tensor.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.25     start
#
#
# description: 
#   Generate the tensor informatino of the cubed-sphere
#   D   : transformation matrix [-1,1] -> [alpha,beta] -> [lon,lat]
#   Dvv : derivative matrix
#   jac : Jacobian
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from numpy import pi, sin, cos, tan, sqrt, fabs, dot
from numpy.testing import assert_array_almost_equal as aa_equal

from pkg.util.quadrature import gausslobatto, legendre




class CubeTensor(object):
    def __init__(self, cubegrid, rotated=False, homme_style=False):
        self.cubegrid = cubegrid
        self.rotated= rotated

        ngq = cubegrid.ngq
        local_ep_size = cubegrid.local_ep_size
        gq_pts, gq_wts = gausslobatto(ngq-1)    # ngq-1: polynomial order

        Dvv = np.zeros(ngq*ngq, 'f8')
        Dinv = np.zeros(local_ep_size*2*2, 'f8')
        jac = np.zeros(local_ep_size, 'f8')


        self.init_derivative_matrix(Dvv, gq_pts)

        if homme_style:
            self.init_transform_matrix_homme(Dinv, jac)
            area_ratio = self.calc_area_ratio(gq_wts, jac)

            Dinv[:] /= np.sqrt(area_ratio)
            jac[:] *= area_ratio

        else:
            self.init_transform_matrix(Dinv, jac)
            #area_ratio1 = self.calc_area_ratio(gq_wts, jac)

            self.revise_transform_matrix_by_elem_area(gq_wts, jac, Dinv)

        #area_ratio2 = self.calc_area_ratio(gq_wts, jac)
        #print cubegrid.myrank, area_ratio1, area_ratio2



        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.gq_pts = gq_pts
        self.gq_wts = gq_wts
        self.Dvv = Dvv
        self.Dinv = Dinv
        self.jac = jac



    def init_derivative_matrix(self, Dvv, gq_pts):
        #-----------------------------------------------------
        # Set the derivative matrix
        # Dvv: derivative matrix
        #-----------------------------------------------------
        ngq = self.cubegrid.ngq
        p_order = ngq - 1

        for idx in xrange(ngq*ngq):
            # originally i: outmost order, j: inmost order
            # but, for convinient, I switch them to get DvvT directly.
            i = idx//ngq
            j = idx%ngq

            if i != j:
                Dvv[idx] = 1/(gq_pts[i] - gq_pts[j]) * \
                        ( legendre(p_order,gq_pts[i]) / legendre(p_order,gq_pts[j]) )

            else:
                if i == 0:
                    Dvv[idx] = - p_order*(p_order+1)/4

                elif i == p_order:
                    Dvv[idx] = p_order*(p_order+1)/4

                elif 0 < i < p_order:
                    Dvv[idx] = 0



    def init_transform_matrix_homme(self, Dinv, jac):
        #-----------------------------------------------------
        # Set the transform matrix
        # D    : transformation matrix [-1,1] -> [alpha,beta] -> [lon,lat]
        # Dinv : inverse matrix of D
        # jac  : Jacobian
        #-----------------------------------------------------
        cubegrid = self.cubegrid

        ne = cubegrid.ne
        local_gq_indices = cubegrid.local_gq_indices    # (local_ep_size,5)
        local_alpha_betas = cubegrid.local_alpha_betas  # (local_ep_size,2)

        for seq, (alpha, beta) in enumerate(local_alpha_betas):
            panel, ei, ej, gi, gj = local_gq_indices[seq]

            # Transform [alpha,beta] -> [lon,lat]
            d = sqrt(tan(alpha)**2 + tan(beta)**2)
            r = sqrt(1 + tan(alpha)**2 + tan(beta)**2)

            if panel in [1,2,3,4]:
                a = 1/(r*cos(alpha))
                b = 0
                c = -tan(alpha)*tan(beta)/(r*r*cos(alpha))
                d = 1/(r*r*cos(alpha)*cos(beta)*cos(beta))

            elif panel in [5,6]:
                if d < 1e-9:
                    a, b, c, d = 1, 0, 0, 1
                else:
                    a = tan(beta)/(d*r*cos(alpha)*cos(alpha))
                    b = -tan(alpha)/(d*r*cos(beta)*cos(beta))
                    c = tan(alpha)/(d*r*r*cos(alpha)*cos(alpha))
                    d = tan(beta)/(d*r*r*cos(beta)*cos(beta))

                    if panel == 6:
                        a, b, c, d = -a, -b, -c, -d

            # Apply the transform [-1,1] -> [alpha,beta]
            a = a*pi/(4*ne)
            b = b*pi/(4*ne)
            c = c*pi/(4*ne)
            d = d*pi/(4*ne)

            '''
            # Transform matrix D [-1,1] -> [alpha,beta] -> [lon,lat]
            D[seq*4+0] = a
            D[seq*4+1] = b
            D[seq*4+2] = c
            D[seq*4+3] = d
            '''

            # Inverse matrix of D
            det = a*d-b*c
            com = 1/det
            Dinv[seq*4+0] =   com*d
            Dinv[seq*4+1] = - com*b
            Dinv[seq*4+2] = - com*c
            Dinv[seq*4+3] =   com*a

            # Jacobian
            jac[seq] = fabs(det)



    def init_transform_matrix(self, Dinv, jac):
        #-----------------------------------------------------
        # Set the transform matrix with a rotational cubed-sphere
        # D    : transformation matrix [-1,1] -> [alpha,beta] -> [lon,lat]
        # Dinv : inverse matrix of D
        # jac  : Jacobian
        #-----------------------------------------------------
        cubegrid = self.cubegrid

        ne = cubegrid.ne
        lat0, lon0 = cubegrid.lat0, cubegrid.lon0       # rotational angle
        local_ep_size = cubegrid.local_ep_size
        local_gq_indices = cubegrid.local_gq_indices    # (local_ep_size,5)
        local_latlons = cubegrid.local_latlons          # (local_ep_size,2)
        local_alpha_betas = cubegrid.local_alpha_betas  # (local_ep_size,2)

        for seq in xrange(local_ep_size):
            panel, ei, ej, gi, gj = local_gq_indices[seq]
            lat, lon = local_latlons[seq]
            alpha, beta = local_alpha_betas[seq]


            #---------------------------------------------
            # computational domain [x1',x2'] -> [x1,x2]
            #---------------------------------------------
            D_comp = (pi/(4*ne))*np.identity(2)


            #---------------------------------------------
            # contravariant [x1,x2] -> [x,y,z]
            #---------------------------------------------
            ta = tan(alpha)
            tb = tan(beta)
            sa2 = 1 + ta*ta     # sec(alpha)**2
            sb2 = 1 + tb*tb     # sec(beta)**2
            rrho3 = ( 1/sqrt(1 + ta*ta + tb*tb) )**3

            if panel == 1:
                D_cs2xyz = np.array( \
                        [[   -ta*sa2,    -tb*sb2], \
                         [   sa2*sb2, -ta*tb*sb2], \
                         [-ta*sa2*tb,    sa2*sb2]] )

            elif panel == 2:
                D_cs2xyz = np.array( \
                        [[  -sa2*sb2,  ta*tb*sb2], \
                         [   -ta*sa2,    -tb*sb2], \
                         [-ta*sa2*tb,    sa2*sb2]] )

            elif panel == 3:
                D_cs2xyz = np.array( \
                        [[    ta*sa2,     tb*sb2], \
                         [  -sa2*sb2,  ta*tb*sb2], \
                         [-ta*sa2*tb,    sa2*sb2]] )

            elif panel == 4:
                D_cs2xyz = np.array( \
                        [[   sa2*sb2, -ta*tb*sb2], \
                         [    ta*sa2,     tb*sb2], \
                         [-ta*sa2*tb,    sa2*sb2]] )

            elif panel == 5:
                D_cs2xyz = np.array( \
                        [[-ta*sa2*tb,    sa2*sb2], \
                         [   sa2*sb2, -ta*tb*sb2], \
                         [    ta*sa2,     tb*sb2]] )

            elif panel == 6:
                D_cs2xyz = np.array( \
                        [[ ta*sa2*tb,   -sa2*sb2], \
                         [   sa2*sb2, -ta*tb*sb2], \
                         [   -ta*sa2,    -tb*sb2]] )


            #---------------------------------------------
            # rotation [x,y,z] -> [x',y',z']
            #---------------------------------------------
            D_rot = np.array( \
                    [[cos(lat0)*cos(lon0), -sin(lon0), -sin(lat0)*cos(lon0)], \
                     [cos(lat0)*sin(lon0),  cos(lon0), -sin(lat0)*sin(lon0)], \
                     [sin(lat0)          ,          0,  cos(lat0)          ]] )


            #---------------------------------------------
            # [x',y',z'] -> [lon,lat]
            #---------------------------------------------
            D_xyz2ll = np.array( \
                    [[         -sin(lon),           cos(lon),        0], \
                     [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)], \
                     [ cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat)]] )


            #---------------------------------------------
            # Transform matrix
            # [-1,1] -> cubed-sphere -> Cartesian -> rotated Cartesian -> latlon
            # [x1',x2'] -> [x1,x2] -> [x,y,z] -> [x',y',z'] -> [lon,lat]
            #---------------------------------------------
            D = dot(D_xyz2ll, dot(D_rot, dot(rrho3*D_cs2xyz, D_comp)))
            aa_equal(D[2,:], [0,0], 15)

            a, b, c, d = D[0,0], D[0,1], D[1,0], D[1,1]

            # Inverse matrix of D
            det = a*d-b*c
            com = 1/det
            Dinv[seq*4+0] =   com*d
            Dinv[seq*4+1] = - com*b
            Dinv[seq*4+2] = - com*c
            Dinv[seq*4+3] =   com*a


            #---------------------------------------------
            # Jacobian
            #---------------------------------------------
            jac[seq] = fabs(det)



    def calc_area_ratio(self, gq_wts, jac):
        from math import fsum, pi
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        local_ep_size = self.cubegrid.local_ep_size
        local_gq_indices = self.cubegrid.local_gq_indices

        elem_areas = np.zeros(local_ep_size, 'f8')
        for seq in xrange(local_ep_size):
            panel, ei, ej, gi, gj = local_gq_indices[seq]
            elem_areas[seq] = jac[seq]*gq_wts[gi-1]*gq_wts[gj-1]
        local_area = fsum(elem_areas)

        local_areas = comm.allgather(local_area)
        area = fsum(local_areas)
        area_ratio = 4*pi/area

        return area_ratio



    def area_spherical_triangle(self, xyz1, xyz2, xyz3):
        '''
        Area of spherical triangle using Girard theorem
        '''
        # Get angles
        cross = np.linalg.norm( np.cross(xyz1, xyz2) )
        dot = np.dot(xyz1, xyz2)
        a = np.arctan2(cross, dot)

        cross = np.linalg.norm( np.cross(xyz2, xyz3) )
        dot = np.dot(xyz2, xyz3)
        b = np.arctan2(cross, dot)

        cross = np.linalg.norm( np.cross(xyz3, xyz1) )
        dot = np.dot(xyz3, xyz1)
        c = np.arctan2(cross, dot)

        s = 0.5*(a+b+c)


        # Area
        inval = np.tan(0.5*s) * np.tan(0.5*(s-a)) * \
                np.tan(0.5*(s-b)) * np.tan(0.5*(s-c))
        area = 4*np.arctan( np.sqrt(inval) )


        return area



    def revise_transform_matrix_by_elem_area(self, gq_wts, jac, Dinv):
        from math import fsum

        ngq = self.cubegrid.ngq
        local_ep_size = self.cubegrid.local_ep_size
        local_xyzs = self.cubegrid.local_xyzs
        local_gq_indices = self.cubegrid.local_gq_indices

        for i in xrange(0, local_ep_size, ngq*ngq):
            # analytic area
            xyz1 = local_xyzs[i+0]
            xyz2 = local_xyzs[i+ngq-1]
            xyz3 = local_xyzs[i+ngq*(ngq-1)]
            xyz4 = local_xyzs[i+ngq*ngq-1]

            area1 = self.area_spherical_triangle(xyz1, xyz2, xyz3)
            area2 = self.area_spherical_triangle(xyz2, xyz3, xyz4)

            anal_area = area1 + area2


            # Gauss-quadrature area
            jwsum = np.zeros(ngq*ngq, 'f8')
            for j in xrange(ngq*ngq):
                panel, ei, ej, gi, gj = local_gq_indices[i+j]
                jwsum[j] = jac[i+j]*gq_wts[gi-1]*gq_wts[gj-1]
            gq_area = fsum(jwsum)


            area_ratio = anal_area/gq_area
            Dinv[i*4:(i+ngq*ngq)*4] /= np.sqrt(area_ratio)
            jac[i:i+ngq*ngq] *= area_ratio
