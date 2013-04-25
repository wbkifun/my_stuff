import os

os.system('f2py -c --fcompiler=gnu95 vector_in_out.pyf vector_in_out.f90')
