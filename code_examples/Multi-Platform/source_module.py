#------------------------------------------------------------------------------
# filename  : source_module.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: System Configuration Team, KIAPS
# update    : 2015.3.18    start
#             2015.9.23    modify the case of CPU-C with f2py
#             2015.10.27   add make_signature_f90() and make_signature_c()
#             2016.3.10    add key-value 'float':'real(4)' in make_signature_c()
#             2016.8.29    remove src string and tempfile
#                          rename get_module() -> compile_using_f2py()
#                          convert ''%() -> ''.format()
#                          add get_module_from_file()
#             2016.8.30    remove logger
#             2016.9.5     modify arguments in the compile_using_f2py()
#
#
# description:
#   Generate the Python module form Fortran 90/95 or C code
#------------------------------------------------------------------------------

import numpy as np
import subprocess as subp
import parse
import os
import glob
import shutil
from importlib.util import spec_from_file_location, module_from_spec



fcompiler = 'gnu95'     # f2py Fortran, $ f2py -c --help-fcompiler
ccompiler = 'unix'      # f2py C,       $ f2py -c --help-compiler



sig_template = '''
python module $MODNAME
  interface
<CONTENT>
  end interface
end python module
'''
def make_signature_c(src):
    #pc = parse.compile('void {}({}) {\n')

    contents = list()
    lines = list()
    for line in src.split('\n'):
        ll = line.lower()

        result = parse.parse('void {}({}) {', ll)
        if result != None:
            funcname, args = result.fixed
            arg_list = [arg.strip().split(' ') for arg in args.split(',')]
            arg_names = [arg[1].strip('*') for arg in arg_list]
            arg_types = [{'int':'integer', 'float':'real(4)', 'double':'real(8)'}[arg[0]] for arg in arg_list]
            is_array = [arg[0].endswith('*') or arg[1].startswith('*') for arg in arg_list]

            lines.append('    subroutine {}({})'.format(funcname, ', '.join(arg_names)))
            lines.append('      intent(c) :: {}'.format(funcname))
            lines.append('      intent(c)')

            for seq, (arg_name, arg_type) in enumerate(zip(arg_names,arg_types)):
                if is_array[seq]:
                    lines.append('      {}, dimension(SIZE{}), intent(IO{}) :: {}'.format(arg_type, arg_name, arg_name, arg_name))
                elif arg_type == 'integer':
                    lines.append('      integer, required, intent(in) :: {}'.format(arg_name))
                else:
                    lines.append('      {}, intent(in) :: {}'.format(arg_type, arg_name))

            lines.append('    end subroutine')
            content = '\n'.join(lines)
            count_replace = 0

        if '::' in line:
            result2 = parse.parse(' {} :: {}, {}', ll.split('//')[-1])
            if result2 != None:
                name, size, intent = result2.fixed
                assert intent in ['in', 'inout']
                content = content.replace('SIZE{}'.format(name), size)
                content = content.replace('IO{}'.format(name), intent)

                count_replace += 1
                if count_replace == np.count_nonzero(is_array):
                    contents.append(content)
                    lines = list()

    assert len(contents) > 0, 'You should specify size and intent information of array arguments in your C code.\n{}'.format(src)

    sig = sig_template.replace('<CONTENT>', '\n'.join(contents))

    return sig




def compile_using_f2py(src_fpath, compiler, flags='', opt_flags='', objs=[]):
    # src : Fortran source file (*.f90)
    # pyf : f2py signature file (*.pyf)
    # compiler : gnu or intel

    dpath = os.path.dirname(src_fpath)
    build_dpath = os.path.join(dpath, 'build')
    if not os.path.exists(build_dpath): os.mkdir(build_dpath)
    os.chdir(build_dpath)

    src_name, code_type = os.path.basename(src_fpath).split('.')
    assert code_type in ['f90','c'], "src_fpath: {}\nError: wrong code_type={}, it should be 'f90' or 'c'".format(src_fpath, code_type)
    mod_name = src_name
    pyf_fname = src_name+'.'+code_type+'.pyf'


    #
    # Signiture file
    #
    if code_type == 'f90':
        cmd = 'f2py --overwrite-signature -h {} -m {} {}'.format(pyf_fname, mod_name, src_fpath)
        ps = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
        stdout, stderr = ps.communicate()
        assert len(stderr) == 0, "{}\n{}".format(stdout.decode('utf-8'), stderr.decode('utf-8'))

        cmd = 'sed -i s/optional/required/g {}'.format(pyf_fname)
        ps = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
        stdout, stderr = ps.communicate()
        assert len(stderr) == 0, "{}\n{}".format(stdout.decode('utf-8'), stderr.decode('utf-8'))

    elif code_type == 'c':
        with open(src_fpath, 'r') as f: src = f.read()
        pyf = make_signature_c(src)
        pyf = pyf.replace('$MODNAME', mod_name)
        with open(pyf_fname, 'w') as f: f.write(pyf)
    

    #
    # Compile
    #
    obj_fnames = ' '.join([name+'.o' for name in objs])

    if code_type == 'f90':
        fcompiler = {'gnu':'gnu95', 'intel':'intelem'}[compiler.lower()]
        cmd = 'f2py -c --fcompiler={} --f90flags={} --opt={} -I. {} {} {}'.format(fcompiler, flags, opt_flags, pyf_fname, obj_fnames, src_fpath)

    elif code_type == 'c':
        ccompiler = {'gnu':'unix', 'intel':'intelem'}[compiler.lower()]
        cmd = 'f2py -c --compiler={} --f90flags={} --opt={} -I. {} {} {}'.format(ccompiler, flags, opt_flags, pyf_fname, obj_fnames, src_fpath)

    print('[compile]', cmd.replace(src_fpath, os.path.basename(src_fpath)))
    ps = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = ps.communicate()
    assert len(stderr) == 0, "{}\n{}".format(stdout.decode('utf-8'), stderr.decode('utf-8'))


    #
    # Rename the generated object file
    # 
    so_fpath_list = glob.glob('{}.cpython-*.so'.format(mod_name))
    assert len(so_fpath_list) == 1, 'Two or more object files are found. The object file should be one.\n{}'.format(so_fpath_list)
    so_fpath = so_fpath_list[0]
    new_so_name = '{}.{}.so'.format(mod_name, code_type)
    shutil.move(so_fpath, new_so_name)




def get_module_from_file(build_dpath, mod_name, code_type):
    so_fpath = os.path.join(build_dpath, '{}.{}.so'.format(mod_name, code_type))
    assert os.path.exists(so_fpath), "{} is not found.".format(so_fpath)

    spec = spec_from_file_location(mod_name, so_fpath)
    return module_from_spec(spec)
