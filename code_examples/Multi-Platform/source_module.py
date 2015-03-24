#------------------------------------------------------------------------------
# filename  : source_module.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
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



def get_module_f90(src):
    # generate a temporary file
    dpath = tempfile.gettempdir()
    tfile = tempfile.NamedTemporaryFile(suffix='.f90', dir=dpath, delete=False)
    tfile.write(src)
    tfile.close()
    

    # paths
    src_path = tfile.name
    mod_path = tfile.name.replace('.f90', '.so')
    mod_name = tfile.name.replace('.f90', '').split('/')[-1]


    # compile
    cmd = 'f2py -c --fcompiler=gnu95 -m %s %s' % (mod_name, src_path)
    #print cmd
    ps = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = ps.communicate()
    assert stderr == '', '%s\n\n%s'%(stdout, stderr)


    # move the so file to the temporary directory
    shutil.move(mod_name+'.so', mod_path)


    # remove when the program is terminated
    atexit.register(os.remove, mod_path)
    atexit.register(os.remove, src_path)


    # return the generated module
    sys.path.append(dpath)
    return __import__(mod_name)




def get_module_c(src):
    # generate a temporary file
    dpath = tempfile.gettempdir()
    tfile = tempfile.NamedTemporaryFile(suffix='.c', dir=dpath, delete=False)
    tfile.write(src)
    tfile.close()
    

    # paths
    src_path = tfile.name
    o_path = tfile.name.replace('.c', '.o')
    so_path = tfile.name.replace('.c', '.so')
    mod_name = tfile.name.replace('.c', '').split('/')[-1]


    # exchange module name in the source code
    f = open(src_path, 'w')
    f.write(src.replace('$MODNAME', mod_name))
    f.close()


    # compile
    cmd1 = 'gcc -O3 -fPIC -g -I/usr/include/python2.7 -c %s -o %s' % (src_path, o_path)
    #print cmd1
    ps = subp.Popen(cmd1.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = ps.communicate()
    assert stderr == '', '%s\n\n%s'%(stdout, stderr)

    cmd2 = 'gcc -shared -o %s %s' % (so_path, o_path)
    #print cmd2
    ps = subp.Popen(cmd2.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = ps.communicate()
    assert stderr == '', '%s\n\n%s'%(stdout, stderr)


    # remove when the program is terminated
    atexit.register(os.remove, o_path)
    atexit.register(os.remove, so_path)
    atexit.register(os.remove, src_path)


    # return the generated module
    sys.path.append(dpath)
    return __import__(mod_name)
