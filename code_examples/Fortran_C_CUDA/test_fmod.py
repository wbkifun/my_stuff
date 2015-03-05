from source_module_f90 import get_module_f90
src_f90 = open('./testmod.F90').read()
mod = get_module_f90(src_f90)

mod.testmod.testsub()
print mod.testmod.testfunc(2)
