#------------------------------------------------------------------------------
# filename  : source_module.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.9.23    modify the case of CPU-C with f2py
#
#
# description:
#   Generate the Python module form Fortran 90/95 or C code
#------------------------------------------------------------------------------

import atexit
import os
import shutil
import subprocess as subp
import sys
import tempfile



fcompiler = 'gnu95'     # f2py Fortran, $ f2py -c --help-fcompiler
compiler = 'unix'       # f2py C, $ f2py -c --help-compiler




def get_module(src, pyf, code_type):
    # src : Fortran source file (*.f90)
    # pyf : f2py signature file (*.pyf)

    assert code_type in ['f90','c']
    sfx = '.f90' if code_type == 'f90' else '.c'

    # generate a temporary file
    dpath = tempfile.gettempdir()
    tmpfile = tempfile.NamedTemporaryFile(suffix=sfx, dir=dpath, delete=False)
    tmpfile.write(src)
    tmpfile.close()


    # paths
    src_path = tmpfile.name
    mod_path = tmpfile.name.replace(sfx, '.so')
    mod_name = tmpfile.name.replace(sfx, '').split('/')[-1]


    # signiture file
    tmpfile_pyf = tempfile.NamedTemporaryFile(suffix='.pyf', dir=dpath, delete=False)
    pyf2 = pyf.replace('$MODNAME', mod_name)
    tmpfile_pyf.write(pyf2)
    tmpfile_pyf.close()
    pyf_path = tmpfile_pyf.name
    

    # compile
    if code_type == 'f90':
        cmd = 'f2py -c --fcompiler=%s %s %s' % (fcompiler, pyf_path, src_path)
    elif code_type == 'c':
        cmd = 'f2py -c --compiler=%s %s %s' % (compiler, pyf_path, src_path)

    #print cmd
    ps = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = ps.communicate()
    assert stderr == '', '%s\n\n%s'%(stdout, stderr)


    # move the so file to the temporary directory
    shutil.move(mod_name+'.so', mod_path)


    # remove when the program is terminated
    atexit.register(os.remove, mod_path)
    atexit.register(os.remove, src_path)
    atexit.register(os.remove, pyf_path)


    # return the generated module
    sys.path.append(dpath)
    return __import__(mod_name)




def get_module_f90(src, pyf):
    return get_module(src, pyf, 'f90')




def get_module_c(src, pyf):
    return get_module(src, pyf, 'c')
