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
    #print mod_name+'.so', mod_path
    shutil.move(mod_name+'.so', mod_path)


    # remove when the program is terminated
    atexit.register(os.remove, mod_path)
    atexit.register(os.remove, src_path)


    # return the generated module
    sys.path.append(dpath)
    return __import__(mod_name)




if __name__ == '__main__':
    import numpy
    from numpy.testing import assert_array_equal as assert_ae


    # setup
    nx = 1000000
    a = numpy.random.rand(nx)
    b = numpy.random.rand(nx)
    c = numpy.zeros(nx)


    #----------------------------------------------------------------------
    # Fortran code
    #----------------------------------------------------------------------
    src_f90 = '''
    SUBROUTINE add(nx, a, b, c)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: nx
      DOUBLE PRECISION, DIMENSION(nx), INTENT(IN) :: a, b
      DOUBLE PRECISION, DIMENSION(nx), INTENT(INOUT) :: c

      INTEGER :: ii

      DO ii=1,nx
        c(ii) = a(ii) + b(ii)
      END DO
    END SUBROUTINE
    '''

    mod = get_module_f90(src_f90)
    add_f90 = mod.add

    add_f90(a, b, c)
    assert_ae(a+b, c)
