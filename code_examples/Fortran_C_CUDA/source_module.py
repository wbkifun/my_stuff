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




if __name__ == '__main__':
    import numpy
    from numpy.testing import assert_array_equal as a_equal


    # setup
    nx = 1000000
    a = numpy.random.rand(nx)
    b = numpy.random.rand(nx)
    c = numpy.zeros(nx)


    #----------------------------------------------------------------------
    # Fortran code
    #----------------------------------------------------------------------
    src_f = '''
    SUBROUTINE add(nx, a, b, c)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: nx
      REAL(8), DIMENSION(nx), INTENT(IN) :: a, b
      REAL(8), DIMENSION(nx), INTENT(INOUT) :: c

      INTEGER :: ii

      DO ii=1,nx
        c(ii) = a(ii) + b(ii)
      END DO
    END SUBROUTINE
    '''

    mod_f = get_module_f90(src_f)
    add_f = mod_f.add

    add_f(a, b, c)
    a_equal(a+b, c)



    #----------------------------------------------------------------------
    # C code
    #----------------------------------------------------------------------
    src_c = '''
    #include <Python.h>
    #include <numpy/arrayobject.h>

    static PyObject *add(PyObject *self, PyObject *args) {
        PyArrayObject *A, *B, *C;
        if (!PyArg_ParseTuple(args, "OOO", &A, &B, &C)) return NULL;

        int nx, i;
        double *a, *b, *c;

        nx = (int)(A->dimensions)[0];
        a = (double*)(A->data);
        b = (double*)(B->data);
        c = (double*)(C->data);

        for (i=0; i<nx; i++) {
            c[i] = a[i] + b[i];
        }

        Py_INCREF(Py_None);
        return Py_None;
    }

    static PyMethodDef ufunc_methods[] = {
        {"add", add, METH_VARARGS, ""},
        {NULL, NULL, 0, NULL}
    };

    PyMODINIT_FUNC init$MODNAME() {
        Py_InitModule("$MODNAME", ufunc_methods);
        import_array();
    }
    '''

    mod_c = get_module_c(src_c)
    add_c = mod_c.add

    add_c(a, b, c)
    a_equal(a+b, c)
