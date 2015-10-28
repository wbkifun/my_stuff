#------------------------------------------------------------------------------
# filename  : source_module.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.9.23    modify the case of CPU-C with f2py
#             2015.10.27   add make_signature_f90() and make_signature_c()
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
import parse
import numpy as np



fcompiler = 'gnu95'     # f2py Fortran, $ f2py -c --help-fcompiler
compiler = 'unix'       # f2py C, $ f2py -c --help-compiler



sig_template = '''
python module $MODNAME
  interface
CONTENT
  end interface
end python module
'''



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




def make_signature_f90(src):
    lines = list()
    for line in src.split('\n'):
        ll = line.lower()
        if 'subroutine' in ll or 'intent(' in ll:
            ll = ll.replace('integer, ', 'integer, required, ')
            lines.append(' '*4 + ll)
        
    sig = sig_template.replace('CONTENT', '\n'.join(lines))

    return sig




def make_signature_c(src):
    pc = parse.compile('void {}({}) {\n')

    contents = list()
    lines = list()
    for line in src.split('\n'):
        ll = line.lower()

        result = parse.parse('void {}({}) {', ll)
        if result != None:
            funcname, args = result.fixed
            arg_list = [arg.strip().split(' ') for arg in args.split(',')]
            arg_names = [arg[1].strip('*') for arg in arg_list]
            arg_types = [{'int':'integer', 'double':'real(8)'}[arg[0]] \
                    for arg in arg_list]
            is_array = [arg[0].endswith('*') or arg[1].startswith('*') \
                    for arg in arg_list]

            lines.append('    subroutine %s(%s)'%(funcname, ', '.join(arg_names)))
            lines.append('      intent(c) :: %s'%(funcname))
            lines.append('      intent(c)')

            for seq, (arg_name, arg_type) in enumerate(zip(arg_names,arg_types)):
                if is_array[seq]:
                    lines.append('      %s, dimension(SIZE%s), intent(IO%s) :: %s'%(arg_type, arg_name, arg_name, arg_name))
                elif arg_type == 'integer':
                    lines.append('      integer, required, intent(in) :: %s'%(arg_name))
                else:
                    lines.append('      %s, intent(in) :: %s'%(arg_type, arg_name))

            lines.append('    end subroutine')
            content = '\n'.join(lines)
            count_replace = 0

        if '::' in line:
            result2 = parse.parse('    // {} :: {}, {}', ll)
            if result2 != None:
                name, size, io = result2.fixed
                assert io in ['in', 'inout']
                content = content.replace('SIZE%s'%(name), size)
                content = content.replace('IO%s'%(name), io)

                count_replace += 1
                if count_replace == np.count_nonzero(is_array):
                    contents.append(content)
                    lines = list()

    sig = sig_template.replace('CONTENT', '\n'.join(contents))

    return sig
