import os

os.system('f2py -m simple_ice_model_advanced -h simple_ice_model_advanced.pyf ../simple_ice_model_modules.f90 simple_ice_model_advanced.f90')

