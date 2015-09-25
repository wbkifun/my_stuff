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


fname = __file__.split('/')[-1]
fdir = __file__.rstrip(fname)




class CubeTensor(object):
    def __init__(self, ne, ngq):
        self.ne = ne
        self.ngq = ngq

        
        #-----------------------------------------------------
        # Read the grid and sparse matrix information
        #-----------------------------------------------------
        cs_fpath = fdir + 'cs_grid_ne%dngq%d.nc'%(ne, ngq)
        cs_ncf = nc.Dataset(cs_fpath, 'r', format='NETCDF4')
        uvp_size = len( cs_ncf.dimensions['uvp_size'] )
        gq_indices = cs_ncf.variables['gq_indices'][:]
        gids = cs_ncf.variables['gids'][:]                  # uvp_size
        alpha_betas = cs_ncf.variables['alpha_betas'][:]    # uvp_size


        #-----------------------------------------------------
        # Set the transform matrix
        # A  : transformation matrix [-1,1] -> [alpha,beta] -> [lon,lat]
        # J  : Jacobian
        #-----------------------------------------------------
        A = np.zeros((uvp_size,2,2), 'f8')
        J = np.zeros(uvp_size, 'f8')

        for u_seq, (alpha, beta) in enumerate(alpha_betas):
            gseq = gids[u_seq]
            panel, ei, ej, gi, gj = gq_indices[gseq]

            # transform [alpha,beta] -> [lon,lat]
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

            # apply the transform [-1,1] -> [alpha,beta]
            a = a*pi/(4*ne)
            b = b*pi/(4*ne)
            c = c*pi/(4*ne)
            d = d*pi/(4*ne)

            A[u_seq,0,0] = a
            A[u_seq,0,1] = b
            A[u_seq,1,0] = c
            A[u_seq,1,1] = d

            # jacobian
            det = a*d-b*c
            J[u_seq] = fabs(det)


        #-----------------------------------------------------
        # Set the derivative matrix
        # dvv: derivative matrix
        #-----------------------------------------------------
        p_order = ngq - 1
        gq_pts, gq_wts = gausslobatto(p_order)
        dvv = np.zeros((ngq,ngq), 'f8')

        for i in xrange(ngq):
            for j in xrange(ngq):
                if i != j:
                    dvv[i,j] = 1/(gq_pts[i] - gq_pts[j]) * \
                            ( legendre(p_order,gq_pts[i]) / legendre(p_order,gq_pts[j]) )

                else:
                    if i == 0:
                        dvv[i,j] = - p_order*(p_order+1)/4

                    elif i == p_order:
                        dvv[i,j] = p_order*(p_order+1)/4

                    elif 0 < i < p_order:
                        dvv[i,j] = 0


        #-----------------------------------------------------
        # Public variables
        #-----------------------------------------------------
        self.uvp_size = uvp_size
        self.A = A
        self.J = J
        self.dvv = dvv
        self.gq_pts = gq_pts
        self.gq_wts = gq_wts



    def save_netcdf(self):
        ne, ngq = self.ne, self.ngq
        uvp_size = self.uvp_size

        ncf = nc.Dataset('cs_tensor_ne%dngq%d.nc'%(ne,ngq), 'w', format='NETCDF4')
        ncf.description = 'Transform matrix, Jacobian, Derivative matrix for the spectral element method on the cubed-Sphere'
        ncf.createDimension('ne', ne)
        ncf.createDimension('ngq', ngq)
        ncf.createDimension('uvp_size', uvp_size)
        ncf.createDimension('2', 2)

        vA = ncf.createVariable('A', 'f8', ('uvp_size','2','2'))
        vJ = ncf.createVariable('J', 'f8', ('uvp_size',))
        vdvv = ncf.createVariable('dvv', 'f8', ('ngq','ngq'))
        vgq_pts = ncf.createVariable('gq_pts', 'f8', ('ngq',))
        vgq_wts = ncf.createVariable('gq_wts', 'f8', ('ngq',))

        vA[:] = self.A
        vJ[:] = self.J
        vdvv[:] = self.dvv
        vgq_pts[:] = self.gq_pts
        vgq_wts[:] = self.gq_wts

        ncf.close()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ne', type=int, help='number of elements')
    parser.add_argument('ngq', type=int, help='number of Gauss qudrature points')
    args = parser.parse_args()
    print 'ne=%d, ngq=%d'%(args.ne, args.ngq) 

    ct = CubeTensor(args.ne, args.ngq)
    ct.save_netcdf()
