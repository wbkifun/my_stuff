import os

os.system('f2py -c --fcompiler=gnu95 simple_ice_model_intermediate.pyf ../simple_ice_model_modules.f90 simple_ice_model_intermediate.f90')
