import os

os.system('f2py -c --fcompiler=gnu95 simple_ice_model_basic.pyf ../simple_ice_model_modules.f90 simple_ice_model_basic.f90')

