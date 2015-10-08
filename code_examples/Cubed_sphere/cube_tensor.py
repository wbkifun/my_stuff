#------------------------------------------------------------------------------
# filename  : cube_tensor.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.25     start
#
#
# description: 
#   Generate the tensor informatino of the cubed-sphere
#   A  : transformation matrix [-1,1] -> [alpha,beta] -> [lon,lat]
#   J  : Jacobian
#   dvv: derivative matrix
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
from numpy import pi, sqrt, tan, cos, fabs

from pkg.util.quadrature import gausslobatto, legendre




class CubeTensor(object):
    def __init__(self, cubegrid):
        ne = cubegrid.ne
        ngq = cubegrid.ngq
        local_ep_size = cubegrid.local_ep_size
        local_gq_indices = cubegrid.local_gq_indices    # (local_ep_size,5)
        local_alpha_betas = cubegrid.local_alpha_betas  # (local_ep_size,2)

        
        #-----------------------------------------------------
        # Set the transform matrix
        # A  : transformation matrix [-1,1] -> [alpha,beta] -> [lon,lat]
        # AI : inverse matrix of A
        # J  : Jacobian
        #-----------------------------------------------------
        AI = np.zeros(local_ep_size*2*2, 'f8')
        J = np.zeros(local_ep_size, 'f8')

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
            # Transform matrix A [-1,1] -> [alpha,beta] -> [lon,lat]
            A[seq*4+0] = a
            A[seq*4+1] = b
            A[seq*4+2] = c
            A[seq*4+3] = d
            '''

            # Inverse matrix of A
            det = a*d-b*c
            com = 1/det
            AI[seq*4+0] =   com*d
            AI[seq*4+1] = - com*b
            AI[seq*4+2] = - com*c
            AI[seq*4+3] =   com*a

            # Jacobian
            J[seq] = fabs(det)


        #-----------------------------------------------------
        # Set the derivative matrix
        # dvv: derivative matrix
        #-----------------------------------------------------
        p_order = ngq - 1
        gq_pts, gq_wts = gausslobatto(p_order)
        dvvT = np.zeros(ngq*ngq, 'f8')

        for idx in xrange(ngq*ngq):
            # originally i: inmost order, j: outmost order
            # but, for convinient, I switch them to get dvvT directly.
            i = idx//ngq    # inmost order -> outmost
            j = idx%ngq     # outmost order -> inmost

            if i != j:
                dvvT[idx] = 1/(gq_pts[i] - gq_pts[j]) * \
                        ( legendre(p_order,gq_pts[i]) / legendre(p_order,gq_pts[j]) )

            else:
                if i == 0:
                    dvvT[idx] = - p_order*(p_order+1)/4

                elif i == p_order:
                    dvvT[idx] = p_order*(p_order+1)/4

                elif 0 < i < p_order:
                    dvvT[idx] = 0


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.local_ep_size = local_ep_size
        self.AI = AI
        self.J = J
        self.dvvT = dvvT
        self.gq_pts = gq_pts
        self.gq_wts = gq_wts
