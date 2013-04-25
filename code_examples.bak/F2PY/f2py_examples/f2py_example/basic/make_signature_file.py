import os

os.system('f2py -m simple_ice_model_basic -h simple_ice_model_basic.pyf ../simple_ice_model_modules.f90 simple_ice_model_basic.f90')

