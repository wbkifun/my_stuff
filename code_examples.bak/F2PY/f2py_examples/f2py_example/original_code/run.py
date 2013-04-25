import numpy as N
import pylab as P
import os
from py_fortran_tools import *

data_output_file = 'tmp.dat'
f_output_file = 'simple_ice_model'
f_input_files = ['simple_ice_model_modules.f90', 'simple_ice_model.f90']

compile_and_run(f_output_file, f_input_files, data_output_file)
# get the data
dict_ = parse_output(data_output_file)

# plot it
shape_ = dict_['thick'].shape
P.plot(dict_['x']/1e3, dict_['thick'][-1,:]+dict_['bed'], label='At $t_{final}$')
P.hold(True)
P.plot(dict_['x']/1e3, dict_['thick'][N.floor(shape_[0]/2),:]+dict_['bed']
       , label='at $t_{final}$/2')
P.plot(dict_['x']/1e3, dict_['thick'][-100,:]+dict_['bed'],
       label='100 time steps before $t_{final}$')
P.xlabel('x (km)')
P.ylabel('b+H (m)')
P.legend()
P.show()
