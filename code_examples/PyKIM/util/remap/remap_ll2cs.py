#!/usr/bin/env python

#------------------------------------------------------------------------------
# filename  : remap_ll2cs.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.18     start
#             2016.1.22     parse a YAML config file 
#
#
# Description: 
#   Remapping Latlon -> Cubed-sphere
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import netCDF4 as nc
import pygrib
import argparse
import yaml

from util.remap.cube_remap import CubeRemap
from numpy.testing import assert_array_equal as a_equal




def get_val(key, subsection, section, defaults=None):
    if subsection != None and subsection.has_key(key):
        return subsection[key]

    elif section != None and section.has_key(key):
        return section[key]

    elif defaults != None and defaults.has_key(key):
        return defaults[key]

    else:
        print "Error: There is not '%s' in %s or %s"%(key, sname, fname)
        sys.exit()




def get_copy_val(key, subsection, section):
    if subsection != None and subsection.has_key(key):
        return subsection[key]

    elif subsection != None and subsection.has_key(key.replace('dst_','src_')):
        return subsection[key]

    elif section != None and section.has_key(key):
        return section[key]

    elif section != None and section.has_key(key.replace('dst_','src_'):
        return section[key]

    else:
        print "Error: There is not '%s' in %s or %s"%(key, sname, fname)
        sys.exit()




#-----------------------------------------------------------------------
# Read a config file
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('yml_fpath', type=str, help='YAML file path')
args = parser.parse_args()

with open(args.yml_fpath, 'r') as ymlfile:
    config = yaml.load(ymlfile)

defaults = config.pop('global')
ne = defaults['ne']
ngq = defaults['ngq']
cs_type = defaults['cs_type']
ll_type = defaults['ll_type']
direction = defaults['direction']
remap_matirx_dir = defaults['remap_matrix_dir']

defaults['dst_dir'] = defaults['dst_dir']. \
        replace('<ne>',str(ne)). \
        replace('<method>',defaults['method'])

print '-'*60
print 'Remapping ancillary data to cubed-sphere grid'
print '-'*60
print 'Target resolution: ne%dnp%d (%s)'%(ne,ngq,cs_type)
'''
print 'Remap method: %s'%default_method
print 'Source dir: %s'%default_src_dir
print 'Destination dir: %s'%default_dst_dir
'''

remap_objs = dict()

for sname, section in config.items():
    print ''
    print sname

    src_fnames = section['src_fname']

    if type(src_fnames) == str:
        src_dir = get_val('src_dir',None,section,defaults)
        src_fpath = src_dir+src_fnames
        src_varname = section['src_varname']
        src_shape = section['src_shape']
        dst_varname = get_copy_val('dst_varname',None,section)
        dst_shape = section['dst_shape']
        method = get_val('method',None,section,defaults)

        remap_args = (ne, cs_type, nlat, nlon, ll_type, direction, method, remap_matrix_dir)
        if remap_objs.has_key(remap_args):
            remap_obj = remap_objs[remap_args]
        else:
            remap_obj = CubeRemap(*remap_args)
            remap_objs[remap_args] = remap_obj

        src_ncf = nc.Dataset(src_fpath, 'r')
        for vname in src_varname:
            src_var = src_ncf.variables[vname][:]
            src_var.shape
            dst_var = np.zeros(


    elif type(src_fnames) == list:
        src_fpaths = list()

        src_dir = section.get('src_dir', defaults['src_dir'])
        src_fpaths = [src_dir+fname for fname in src_fnames]
        src_varnames = [section['varname'] for fname in src_fnames]
        src_shapes = [section['shape'] for fname in src_fnames]
        dst_varnames = [section['to_varname'] for fname in src_fnames]
        dst_shapes = [section['shape'] for fname in src_fnames]
        methods = [get_val('method', None, section, defaults) for fname in src_fnames]

    elif type(src_fnames) == dict:
        src_fpaths = list()
        src_varnames = list()
        src_shapes = list()
        methods = list()
        for fname, sub in src_fnames.items():
            src_dir = get_val('src_dir', sub, section, defaults)
            src_fpaths.append(src_dir+fname)
            src_varnames.append( get_val('varname', sub, section, defaults) )
            src_shapes.append( get_val('shape', sub, section) )
            methods.append( get_val('method', sub, section, defaults) )


    dst_dir = defaults['dst_dir']
    dst_fname = section['dst_fname'].replace('<ne>',str(ne)).replace('<ngq>',str(ngq))
    dst_fpath = dst_dir + dst_fname


    print '\t-> %s'%dst_fpath
    print ''

    for i, src_fpath in enumerate(src_fpaths):
        src_varname = src_varnames[i]
        src_shape = src_shapes[i]

        print '\t%s'%(src_fpath)
        print '\t\t%s %s'%(src_varname, src_shape)

        remap_args = (ne, cs_type, nlat, nlon, ll_type, direction, method, remap_matrix_dir)
        remap_obj = CubeRemap(*remap_args)
        remap_objs[remap_args] = remap_obj




    '''
    dst_dir= config.get(section,'dst_basedir')
    src_fnames = config.get(section,'src_fname').split(',')
    src_fnames = [fname.strip('\n') for fname in src_fnames]
    dst_fname = config.get(section,'dst_fname').replace('<cube_res>','ne%dnp4'%ne)

    
    print method
    print src_dir
    print dst_dir
    print src_fnames
    print dst_fname
    '''

"""
defaults = {'method':method, 'src_basedir':src_dir, 'dst_basedir':dst_dir}
config = ConfigParser.SafeConfigParser(defaults)
config.read(args.cfg_fpath)

for section in sections:
    print section
    for options in config.options(section):
        print options, config.get(section,options), str(type(options))

    '''
    method = config.get(section,'method')
    src_dir= config.get(section,'src_basedir')
    dst_dir= config.get(section,'dst_basedir')
    src_fnames = config.get(section,'src_fname').split(',')
    src_fnames = [fname.strip('\n') for fname in src_fnames]
    dst_fname = config.get(section,'dst_fname').replace('<cube_res>','ne%dnp4'%ne)

    
    print method
    print src_dir
    print dst_dir
    print src_fnames
    print dst_fname
    '''
print config.get('Aerosol MACC','src_fname')
"""


#-----------------------------------------------------------------------
# Source data
#-----------------------------------------------------------------------
grbs = pygrib.open('/nas2/user/khkim/clim.snoalb.grib')

grb = grbs.message(1)
data, lats, lons = grb.data()

ll_var = np.zeros_like(data)
ll_var[:,:180] = data[:,180:]
ll_var[:,180:] = data[:,:180]

'''
nlat, nlon = data.shape
src_var = np.zeros(nlat*nlon, 'f8')
for j in xrange(nlat):
    for i in xrange(nlon):
        src_var[j*nlon+i] = ll_var[j,i]

a_equal(src_var, ll_var.ravel())
'''


#-----------------------------------------------------------------------
# Remapping
#-----------------------------------------------------------------------
ne = 30
cs_type = 'regular'
nlat, nlon = data.shape
ll_type = 'regular'
direction = 'll2cs'
method = 'vgecore'

obj = CubeRemap(ne, cs_type, nlat, nlon, ll_type, direction, method, remap_matrix_dir)

up_size = obj.ncf.up_size
cs_var = np.zeros(48602, ll_var.dtype)
obj.remap(ll_var.ravel(), cs_var)


#-----------------------------------------------------------------------
# Save as NetCDF
#-----------------------------------------------------------------------
ncf = nc.Dataset(output_fpath, 'w', format='NETCDF3_CLASSIC')
for attr, val in obj.ncf.__dict__.items():
    setattr(ncf, attr, val)

if direction == 'll2cs':
    ncf.createDimension('up_size', up_size)
    vvar = ncf.createVariable('var', 'f8', ('up_size',))
    vvar[:] = cs_var[:]

else:
    print 'direction=%s is not supported yet'%direction
    sys.exit()

ncf.close()
