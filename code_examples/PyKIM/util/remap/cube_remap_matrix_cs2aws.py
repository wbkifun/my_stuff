#------------------------------------------------------------------------------
# filename  : cube_remap_matrix_cs2cs.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.8.3      start, from cube_remap_matrix_cs2cs.py
#
#
# Description: 
#   Generate a remap_matrix to remap from cubed-sphere to AWS observations
#------------------------------------------------------------------------------

import numpy as np
import netCDF4 as nc
import argparse
import os
import sys
from mpi4py import MPI

from cube_remap_matrix import CubeGridRemap
from util.convert_coord.cart_ll import latlon2xyz



comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myrank = comm.Get_rank()




class AWSVoronoi:
    def __init__(self, aws_fpath):
        import pickle
        with open(aws_fpath, 'rb') as f:
            data = pickle.load(f)
            self.ids = data['ids']
            self.latlons = [(np.deg2rad(lat), np.deg2rad(lon)) for lat,lon in data['latlons']]
            self.voronoi_latlons = data['voronoi_latlons']
            self.nsize = len(self.ids)


    def get_voronoi(self, idx):
        return [latlon2xyz(np.deg2rad(lat),np.deg2rad(lon)) for lat,lon in self.voronoi_latlons[idx]]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_ne', type=int, help='number of elements')
    parser.add_argument('src_type', type=str, help='cube grid type', \
            choices=['regular', 'rotated'])
    parser.add_argument('aws_pkl_fpath', type=str, help='pickle file path of AWS voronoi')
    parser.add_argument('method', type=str, help='remap method', \
            choices=['bilinear', 'vgecore', 'nearest'])
    parser.add_argument('output_dir', nargs='?', type=str, \
            help='output directory', default='./remap_matrix/')
    args = parser.parse_args()


    ngq = 4
    src_ne = args.src_ne
    src_type = args.src_type
    aws_fpath = args.aws_pkl_fpath
    dst_type = 'aws'
    method = args.method
    output_dir = args.output_dir
    output_fname = "remap_cs2aws_ne{}_{}_{}_{}.nc".format(src_ne, src_type, dst_type, method)
    output_fpath = output_dir + output_fname


    #-------------------------------------------------
    # check
    #-------------------------------------------------
    if myrank == 0:
        print("source grid: cubed-sphere({}), ne={}".format(src_type, src_ne))
        print("target : AWS korea")
        print("remap method: {}".format(method))
        print("output directory: {}".format(output_dir))
        print("output filename: {}".format(output_fname))

        #yn = raw_input('Continue (Y/n)? ')
        #if yn.lower() == 'n':
        #    sys.exit()

        if not os.path.exists(output_dir):
            print("{} is not found. Make output directory.".format(output_dir))
            os.makedirs(output_dir)

        #if os.path.exists(output_fpath):
        #    yn = raw_input("%s is found. Overwrite(Y/n)? "%output_fpath)
        #    if yn.lower() == 'n':
        #        sys.exit()

    comm.Barrier()


    if myrank ==0:
        #------------------------------------------------------------
        print("Prepare to save as NetCDF")
        #------------------------------------------------------------
        ncf = nc.Dataset(output_fpath, 'w', format='NETCDF3_CLASSIC')
        ncf.description = 'Remapping between Cubed-sphere and Latlon grids'
        ncf.remap_method = method


    #-------------------------------------------------
    if myrank == 0: print("Make a remap matrix")
    #-------------------------------------------------
    cs_obj = CubeGridRemap(src_ne, ngq, src_type==src_type)
    aws_obj = AWSVoronoi(aws_fpath)

    if method == 'bilinear':
        from cube_remap_bilinear import Bilinear
        rmp = Bilinear(cs_obj, aws_obj, 'cs2aws')
        src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    elif method == 'vgecore':
        from cube_remap_vgecore import VGECoRe
        rmp = VGECoRe(cs_obj, aws_obj, 'cs2aws')
        dst_address, src_address, remap_matrix = rmp.make_remap_matrix_mpi()

    elif method == 'nearest':
        from cube_remap_nearest import Nearest
        rmp = Nearest(cs_obj, aws_obj, 'cs2aws')
        src_address = rmp.make_remap_matrix_mpi()


    if myrank ==0:
        #------------------------------------------------------------
        print("Save as NetCDF")
        #------------------------------------------------------------
        ncf.ngq = ngq
        ncf.src_ne = src_ne
        ncf.src_type = src_type
        ncf.dst_type = dst_type
        ncf.dst_nsize = aws_obj.nsize

        if method == 'bilinear':
            rmp.set_netcdf_remap_matrix(ncf, src_address, remap_matrix)
        elif method == 'vgecore':
            rmp.set_netcdf_remap_matrix(ncf, dst_address, src_address, remap_matrix)
        elif method == 'nearest':
            rmp.set_netcdf_remap_matrix(ncf, src_address)

        ncf.close()
        print("Done.")
