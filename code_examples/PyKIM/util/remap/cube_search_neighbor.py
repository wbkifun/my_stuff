from __future__ import division
import numpy as np
import netCDF4 as nc
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('ne', type=int, help='number of elements')
parser.add_argument('uid', type=int, help='sequence of unique point')
args = parser.parse_args()

cs_grid_dir = '/data/khkim/cs_grid/'

ne, ngq = args.ne, 4
uid = args.uid
ncf = nc.Dataset(cs_grid_dir+'cs_grid_ne%dngq%d.nc'%(ne,ngq), 'r')

gids = ncf.variables['gids'][:]
gq_indices = ncf.variables['gq_indices'][:]

gid = gids[uid]

idx2gids = dict()
for gid, (panel,ei,ej,gi,gj) in enumerate(gq_indices):
    idx2gids[(panel,ei,ej,gi,gj)] = gid

print idx2gids[
not yet
